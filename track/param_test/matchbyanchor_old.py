# 该版本已废弃
# 该脚本用于检测颗粒周围以2倍d50为半径的区域内的颗粒个数，并计算局部布局差异
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import csv
from scipy.spatial import KDTree  # 导入 KDTree
import concurrent.futures
from joblib import Parallel, delayed
import h5py


# 按需读取某个pending颗粒的预测子集
def get_prediction_for_particle_from_h5(h5_path, pending_particle):
    with h5py.File(h5_path, 'r') as h5f:
        particles_grp = h5f['particles']
        pid = str(pending_particle)
        if pid not in particles_grp:
            print(f"Warning: Particle {pending_particle} not found in predictions.")
            return None
        g = particles_grp[pid]
        subsets = []
        for j in range(len(g["subset_predicted_label"])):
            subsets.append({
                "predicted_label": int(g["subset_predicted_label"][j]),
                "probability": g["subset_probability"][j].tolist()
            })
        return {
            "subsets": subsets,
            "final_predicted_label": int(g.attrs["final_predicted_label"]),
            "final_avg_probability": float(g.attrs["final_avg_probability"])
        }


# 并行处理每个pending颗粒的候选查询
def get_candidate_particles(
    pending_particles, ref_mapping, def_mapping, radius, def_sample_height, org_sample_height, pred_h5_path, strain, anchor_ref_labels
):
    """
    并行获取候选颗粒的个数及标签，并计算预测概率（按需读取h5）。
    返回: candidate_info, no_candidate_pending
    """
    candidate_info = {}
    no_candidate_pending = []
    search_radius = radius  # 2*d50

    # 用KDTree加速ref查找
    ref_labels = list(ref_mapping.keys())
    ref_centroids = np.array([ref_mapping[lab]['corrected_centroid(physical)'] for lab in ref_labels])
    ref_kdtree = KDTree(ref_centroids)

    def process_one(pending_particle):
        pend_centroid = def_mapping[pending_particle]['corrected_centroid(physical)']
        relative_z = pend_centroid[2] / def_sample_height
        displacement = np.array([0, 0, strain * relative_z * org_sample_height])
        mapped_position = pend_centroid + displacement

        # 查询2*d50范围内的ref颗粒
        idxs = ref_kdtree.query_ball_point(mapped_position, r=search_radius)
        candidates = []
        # 按需读取预测
        prediction = get_prediction_for_particle_from_h5(pred_h5_path, pending_particle)
        for idx in idxs:
            ref_lab = ref_labels[idx]
            if ref_lab in anchor_ref_labels: # 排除锚点作为候选颗粒
                continue
            ref_centroid = ref_centroids[idx]
            distance = np.linalg.norm(ref_centroid - mapped_position)
            if distance <= search_radius:
                if prediction is not None:
                    subsets = prediction["subsets"]
                    # 注意ref_lab-1可能越界，需保证ref_lab在probability范围内
                    ref_idx = ref_lab - 1
                    ref_probs = []
                    for subset in subsets:
                        prob_arr = subset["probability"]
                        if 0 <= ref_idx < len(prob_arr):
                            ref_probs.append(prob_arr[ref_idx])
                    avg_prob = np.mean(ref_probs) if ref_probs else None
                else:
                    avg_prob = None
                candidates.append({
                    "candidate_label": ref_lab,
                    "distance": distance,
                    "avg_prob": avg_prob
                })
        return pending_particle, candidates

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_one)(pending_particle) for pending_particle in pending_particles
    )
    for pending_particle, candidates in results:
        candidate_info[pending_particle] = candidates
        if len(candidates) == 0:
            no_candidate_pending.append(pending_particle)
    return candidate_info, no_candidate_pending


def build_kdtree(anchor_data):
    """
    构建 KD-Tree，用于快速查找锚点。
    anchor_data: 锚点信息字典，键为 def_label
    返回值: KD-Tree 对象和锚点标签列表
    """
    anchor_labels = list(anchor_data.keys())
    anchor_def_centroids = np.array([anchor_data[label]["def_centroid"] for label in anchor_labels])
    kdtree = KDTree(anchor_def_centroids)
    return kdtree, anchor_labels


def get_nearby_anchors(pending_particle, kdtree, anchor_labels, def_mapping, anchor_radius, K=10, restrict_radius=True):
    """
    第二轮：restrict_radius=True，取radius内（如4d50）最近K个锚点，不足K个则返回空（跳过）。
    第三轮：restrict_radius=False，直接取最近K个已匹配颗粒（锚点+二轮已匹配），不限制半径。
    """
    pending_centroid = def_mapping[pending_particle]['corrected_centroid(physical)']
    if restrict_radius:
        idxs = kdtree.query_ball_point(pending_centroid, r=anchor_radius)
        if len(idxs) < K:
            return []  # 不足K个则跳过
        # 取radius内最近K个
        dists = [np.linalg.norm(kdtree.data[i] - pending_centroid) for i in idxs]
        sorted_idxs = [idx for _, idx in sorted(zip(dists, idxs))]
        chosen = sorted_idxs[:K]
        return [anchor_labels[i] for i in chosen]
    else:
        dists, nearest = kdtree.query(pending_centroid, k=K)
        if np.isscalar(nearest):
            nearest = [nearest]
        return [anchor_labels[i] for i in nearest]


# 计算局部布局差异（LD）
def compute_LD(pending_particle, candidate_labels, nearby_anchor_labels, anchor_data, def_mapping, ref_mapping):
    """
    计算局部布局差异（LD）。
    pending_particle: 待匹配颗粒的标签
    candidate_labels: 候选颗粒的标签列表
    nearby_anchor_labels: 待匹配颗粒周围的锚点标签列表
    anchor_data: 锚点信息字典，键为 def_label
    def_mapping: 变形体系颗粒属性映射
    ref_mapping: 参考体系颗粒属性映射
    返回值: 每个候选颗粒的 LD 误差列表
    """
    # 获取待匹配颗粒的质心
    pending_centroid = def_mapping[pending_particle]['corrected_centroid(physical)']

    # 获取锚点的质心
    anchor_def_centroids = []
    anchor_ref_centroids = []
    for anchor_label in nearby_anchor_labels:
        anchor_info = anchor_data[anchor_label]
        anchor_def_centroids.append(anchor_info["def_centroid"])
        anchor_ref_centroids.append(anchor_info["ref_centroid"])

    anchor_def_centroids = np.array(anchor_def_centroids)  # (N, 3), N为锚点个数
    anchor_ref_centroids = np.array(anchor_ref_centroids)  # (N, 3)

    # 计算待匹配颗粒的相对向量
    delta_pending = anchor_def_centroids - pending_centroid  # (N, 3)

    # 计算每个候选颗粒的 LD
    ld_errors = {}
    for candidate_label in candidate_labels:
        candidate_centroid = ref_mapping[candidate_label]['corrected_centroid(physical)']
        delta_candidate = anchor_ref_centroids - candidate_centroid # (N, 3)
        diff = delta_pending - delta_candidate  # (N, 3)
        ld = np.sqrt(np.mean(np.sum(diff * diff, axis=1))) # np.sum(diff * diff(每个元素自己平方), axis=1)为求2范数(∥∥)(每个元素平方后相加再开根)的平方，
                                                        # axis=1因为每一行是一个锚点的差值向量，第0维是锚点的个数，np.mean求平均值, 为公式中1/M∑
        ld_errors[candidate_label] = ld

    return ld_errors


# 计算位移一致性差异（UD）
def compute_UD(pending_particle, candidate_labels, nearby_anchor_labels, anchor_data, def_mapping, ref_mapping):
    # 收集锚点的位移
    disps = [anchor_data[l]['displacement'] for l in nearby_anchor_labels]
    disps = np.stack(disps)
    median_disp = np.median(disps, axis=0)
    ud_errors = {}
    # 获取待匹配颗粒的质心（def体系）
    pending_centroid = def_mapping[pending_particle]['corrected_centroid(physical)']
    for c in candidate_labels:
        # 候选颗粒的质心（ref体系）
        candidate_centroid = ref_mapping[c]['corrected_centroid(physical)']
        u = pending_centroid - candidate_centroid
        ud_errors[c] = np.linalg.norm(u - median_disp)
    return ud_errors


# def compute_weighted_deviation_indicators(vals, weight):
#     """
#     根据公式计算 UD或LD 的比值和加权结果。
#     vals: 候选颗粒的 UD 或 LD 值列表
#     weight: UD或LD 的权重 
#     返回值: 包含 UD/LD、比值和加权结果的列表
#     """
#     vmax = max(vals) if vals else 1  # 避免除以 0

#     results = []
#     for v in vals:
#         r = v / vmax  # 计算比值
#         results.append({
#             "value": v,
#             "ratio": r,
#             "weighted": weight*r
#         })
#     return results
def compute_weighted_deviation_indicators(vals, weight):
    """
    根据公式计算 UD或LD 的比值和加权结果。
    vals: 候选颗粒的 UD 或 LD 值列表
    weight: UD或LD 的权重 
    返回值: 包含 UD/LD、比值和加权结果的列表
    """
    # 这个比值版本分母是所有值的总和，而不是最大值
    v_all = np.sum(vals)  # 所有值的总和，用于计算加权
    results = []
    for v in vals:
        r = v / v_all  # 计算比值
        results.append({
            "value": v,
            "ratio": r,
            "weighted": weight*r
        })
    return results

def process_candidates_final_score(
    pending_particles, candidate_info, kdtree, anchor_labels, anchor_data,
    def_mapping, ref_mapping, anchor_radius, K, wUD, wLD, restrict_radius=True
):
    """
    并行处理每个pending颗粒的分数计算。
    返回: results, skipped_pending
    """
    def process_one(p):
        cands = candidate_info.get(p, [])
        if not cands or len(cands) == 0:
            return None  # 已在get_candidate_particles中统计
        nearby = get_nearby_anchors(p, kdtree, anchor_labels, def_mapping, anchor_radius, K, restrict_radius)
        if len(nearby) < K:
            return {"skipped": True, "pending": p, "anchor_count": len(nearby), "candidate_count": len(cands)}
        labels = [c['candidate_label'] for c in cands]
        probs = [c['avg_prob'] for c in cands]
        ud = compute_UD(p, labels, nearby, anchor_data, def_mapping, ref_mapping)
        ld = compute_LD(p, labels, nearby, anchor_data, def_mapping, ref_mapping)
        ud_vals = [ud[lab] for lab in labels]
        ld_vals = [ld[lab] for lab in labels]
        ud_info = compute_weighted_deviation_indicators(ud_vals, wUD)
        ld_info = compute_weighted_deviation_indicators(ld_vals, wLD)
        scores = [ud_info[i]['weighted'] + ld_info[i]['weighted'] + (1 - probs[i])
                  for i in range(len(labels))]
        best = int(np.argmin(scores))
        result_list = []
        for i, lab in enumerate(labels):
            result_list.append({
                'pending': p,
                'candidate': lab,
                'prob': probs[i],
                'UD': ud[lab],
                'UD_ratio': ud_info[i]['ratio'],
                'weighted_UD': ud_info[i]['weighted'],
                'LD': ld[lab],
                'LD_ratio': ld_info[i]['ratio'],
                'weighted_LD': ld_info[i]['weighted'],
                'final_score': scores[i],
                'matched': i == best
            })
        return {"skipped": False, "results": result_list}

    parallel_results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_one)(p) for p in pending_particles
    )
    results = []
    skipped_pending = []
    for item in parallel_results:
        if item is None:
            continue
        if item["skipped"]:
            skipped_pending.append({
                "pending": item["pending"],
                "anchor_count": item["anchor_count"],
                "candidate_count": item["candidate_count"]
            })
        else:
            results.extend(item["results"])
    return results, skipped_pending


def load_properties(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels = data['label']
    centroids = data['corrected_centroid(physical)']
    mapping = {}
    for i, lab in enumerate(labels):
        mapping[lab] = {
            'corrected_centroid(physical)': np.array(centroids[i]),
        }
    return mapping


def plot_histogram(candidate_counts, output_file):
    counts = list(candidate_counts.values())
    total = len(counts)
    percentages = [count / total * 100 for count in counts]

    plt.figure(figsize=(8, 6))
    plt.hist(counts, bins=range(min(counts), max(counts) + 2), weights=percentages, edgecolor='black', alpha=0.7)
    plt.xlabel('number of candidate particles')
    plt.ylabel('percentage (%)')
    plt.title('histogram of candidate particles')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    histogram_path = os.path.splitext(output_file)[0] + '_histogram.png'
    plt.savefig(histogram_path)
    plt.close()
    print(f"直方图已保存到 {histogram_path}")


def main():
    parser = argparse.ArgumentParser(description="检测颗粒周围以2倍d50为半径的区域内的颗粒个数")
    parser.add_argument("--ref_type", required=True, help="参考体系类型，例如 origin, load1, load5, load10, load15")
    parser.add_argument("--def_type", required=True, help="变形体系类型，例如 origin, load1, load5, load10, load15")

    parser.add_argument('--pred_h5', default=None, help="预测结果 h5 文件")
    parser.add_argument('--ref_npz', default=None, help="参考体系颗粒属性 npz 文件")
    parser.add_argument('--def_npz', default=None, help="变形体系颗粒属性 npz 文件")
    parser.add_argument('--init_match_file', default=None, help="初次匹配结果 npz 文件")
    parser.add_argument('--output_dir', default=None, help="输出文件路径")
    parser.add_argument('--d50', type=float, default=0.72, help="D50 值")
    parser.add_argument("--inter_strain", type=float, default=None, help="阶段间应变值（用于映射位置）")
    parser.add_argument("--radius", type=float, default=None, help="候选颗粒搜索半径（默认2×d50）")
    parser.add_argument("--K", type=int, default=10, help="每个待匹配颗粒至少返回的锚点总数")
    args = parser.parse_args()

    # 自动补全文件路径
    if args.pred_h5 is None:
        args.pred_h5 = f"/share/home/202321008879/data/track_results/{args.def_type}to{args.ref_type}/{args.def_type}_ai_pred.h5"
    if args.ref_npz is None:
        args.ref_npz = f"/share/home/202321008879/data/sand_propertities/{args.ref_type}_ai_particle_properties/{args.ref_type}_particle_properties.npz"
    if args.def_npz is None:
        args.def_npz = f"/share/home/202321008879/data/sand_propertities/{args.def_type}_ai_particle_properties/{args.def_type}_particle_properties.npz"
    if args.init_match_file is None:
        args.init_match_file = f"/share/home/202321008879/data/track_results/{args.def_type}to{args.ref_type}/{args.def_type}to{args.ref_type}_initmatch.npz"
    if args.output_dir is None:
        args.output_dir = f"/share/home/202321008879/data/track_results/single_load1toorigin"

    # 自动计算应变差
    strain_mapping = {"origin": 0.0, "load1": 0.01, "load5": 0.05, "load10": 0.1, "load15": 0.15, 'originnew_labbotm1k':0.0,'load1_selpar':0.01}
    if args.inter_strain is None:
        args.inter_strain = strain_mapping[args.def_type] - strain_mapping[args.ref_type]
    # 权重应变值，用于计算权重
    w_strain = strain_mapping[args.def_type]
    print(f"Using inter_strain: {args.inter_strain} (def: {args.def_type} - ref: {args.ref_type})")
    print(f"Using w_strain: {w_strain} (def: {args.def_type})")

    # 自动设置radius
    if args.radius is None:
        args.radius = 2 * args.d50

    org_sample_height = 58.57788148034935

    # 加载初次匹配结果
    init_match_data = np.load(args.init_match_file, allow_pickle=True)
    anchor_particles = init_match_data['anchor_particles'].item()
    pending_particles = init_match_data['pending_particles']
    ref_sample_height = init_match_data['sample_height'].item()  # 只有在load1toorigin中名称才是sample_height，其余都为ref_sample_height
    def_sample_height = init_match_data['def_sample_height'].item()

    # 加载属性数据
    ref_mapping = load_properties(args.ref_npz)
    def_mapping = load_properties(args.def_npz)
    
    # 构建锚点数据为字典形式
    anchor_data = {}
    for def_label, anchor_info in anchor_particles.items():
        ref_label = anchor_info["predicted_reference_label"]
        disp = np.array(anchor_info["displacement(mm)"])
        # 如果disp是标量，补齐为3维；如果是一维数组，直接用
        if disp.ndim == 0:
            disp = np.array([disp, 0, 0])
        elif disp.shape != (3,):
            disp = disp.flatten()
            if disp.shape[0] != 3:
                raise ValueError(f"displacement shape error for label {def_label}: {disp}")
        anchor_data[def_label] = {
            "ref_label": ref_label,
            "def_centroid": def_mapping[def_label]['corrected_centroid(physical)'].tolist(),
            "ref_centroid": ref_mapping[ref_label]['corrected_centroid(physical)'].tolist(),
            "displacement": disp,
        }
    
    # 计算 UD 和 LD 的权重
    weightUD = w_strain / (w_strain + 0.05)  
    weightLD = 1 - weightUD
    print(f"finish loading data.")
    # ========== 第一轮：pointconv筛选锚点 ==========
    # 这里略，假定anchor_particles, pending_particles已分好

    # ========== 第二轮：4d50内锚点，至少10个，否则跳过 ==========
    print(f"开始第二轮匹配...")
    radius_4d50 = 4 * args.d50
    # 构建锚点KDTree
    kdtree, anchor_labels = build_kdtree(anchor_data)
    print(f"锚点KDTree构建完成，锚点个数：{len(anchor_labels)}")

    # 锚点的ref标签
    anchor_ref_labels = set([info["predicted_reference_label"] for info in anchor_particles.values()])

    # 新增：不再整体读取预测，只传递h5路径
    pred_h5_path = args.pred_h5

    # 构建候选颗粒信息（排除锚点）
    candidate_info, no_candidate_pending = get_candidate_particles(
        pending_particles=pending_particles,
        ref_mapping=ref_mapping,
        def_mapping=def_mapping,
        radius=args.radius,
        def_sample_height=def_sample_height,
        org_sample_height=org_sample_height,
        pred_h5_path=pred_h5_path,
        strain=args.inter_strain,
        anchor_ref_labels=anchor_ref_labels,
    )
    print(f"候选颗粒信息构建完成，总待匹配颗粒数：{len(pending_particles)}，无候选颗粒数：{len(no_candidate_pending)}")

    # 第二轮匹配
    results_round2, skipped_pending2 = process_candidates_final_score(
        pending_particles=pending_particles,
        candidate_info=candidate_info,
        kdtree=kdtree,
        anchor_labels=anchor_labels,
        anchor_data=anchor_data,
        def_mapping=def_mapping,
        ref_mapping=ref_mapping,
        anchor_radius=radius_4d50, 
        K=args.K,
        wUD=weightUD,
        wLD=weightLD,
        restrict_radius=True
    )
    print(f"第二轮匹配完成，已匹配颗粒数：{len([r for r in results_round2 if r['matched']])}")
    print(f"第二轮无候选颗粒pending数：{len(no_candidate_pending)}，锚点不足pending数：{len(skipped_pending2)}")
    # 统计第二轮已匹配
    matched_round2 = {r['pending']: r['candidate'] for r in results_round2 if r['matched']}
    # 合并锚点和二轮已匹配作为新锚点颗粒，准备第三轮
    anchor_and_matched = dict(anchor_data)
    for p, c in matched_round2.items():
        disp = np.array(def_mapping[p]['corrected_centroid(physical)'] - ref_mapping[c]['corrected_centroid(physical)'])
        if disp.ndim == 0:
            disp = np.array([disp, 0, 0])
        elif disp.shape != (3,):
            disp = disp.flatten()
            if disp.shape[0] != 3:
                raise ValueError(f"displacement shape error for label {p}: {disp}")
        anchor_and_matched[p] = {
            "ref_label": c,
            "def_centroid": def_mapping[p]['corrected_centroid(physical)'].tolist(),
            "ref_centroid": ref_mapping[c]['corrected_centroid(physical)'].tolist(),
            "displacement": disp,
        }
    print(f"锚点和二轮已匹配颗粒合并完成，锚点+已匹配颗粒数：{len(anchor_and_matched)}")

    # ========== 第三轮：最近10个已匹配颗粒 ==========
    kdtree3, anchor_labels3 = build_kdtree(anchor_and_matched)
    print(f"第三轮KDTree构建完成")
    # 所有待匹配颗粒都参与第三轮
    results_round3, skipped_pending3 = process_candidates_final_score(
        pending_particles=pending_particles,
        candidate_info=candidate_info,
        kdtree=kdtree3,
        anchor_labels=anchor_labels3,
        anchor_data=anchor_and_matched,
        def_mapping=def_mapping,
        ref_mapping=ref_mapping,
        anchor_radius=0,  # 不限制半径
        K=args.K,
        wUD=weightUD,
        wLD=weightLD,
        restrict_radius=False
    )
    print(f"第三轮匹配完成，已匹配颗粒数：{len([r for r in results_round3 if r['matched']])}")
    print(f"第三轮锚点不足pending数：{len(skipped_pending3)}")

    # 标记第三轮是否发生修正
    skipped_pending2_labels = {item["pending"] for item in skipped_pending2}  # 第二轮锚点不足的颗粒标签
    best2 = {r['pending']: r['candidate'] for r in results_round2 if r['matched']}
    best3 = {r['pending']: r['candidate'] for r in results_round3 if r['matched']}
    changed = {k: (best2.get(k), v) for k, v in best3.items() 
               if best2.get(k) != v and k not in skipped_pending2_labels}
    
    # ========== 保存结果 ==========
    os.makedirs(args.output_dir, exist_ok=True)
    # 第二轮
    output_file2 = os.path.join(args.output_dir, 'matchbyanchor_results_round2.csv')
    with open(output_file2, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "pending_label", "candidate_label", "avg_prob",
            "UD", "UD_ratio", "weighted_UD", "LD", "LD_ratio", "weighted_LD", "final_score", "matched"
        ])
        for result in results_round2:
            writer.writerow([
                result["pending"],
                result["candidate"],
                f"{result['prob']:.4f}" if result["prob"] is not None else "None",
                f"{result['UD']:.4f}",
                f"{result['UD_ratio']:.4f}",
                f"{result['weighted_UD']:.4f}",
                f"{result['LD']:.4f}",
                f"{result['LD_ratio']:.4f}",
                f"{result['weighted_LD']:.4f}",
                f"{result['final_score']:.4f}",
                result["matched"]
            ])
    print(f"第二轮匹配结果已保存到 {output_file2}")

    # 第三轮
    output_file3 = os.path.join(args.output_dir, 'matchbyanchor_results_round3.csv')
    with open(output_file3, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "pending_label", "candidate_label", "avg_prob",
            "UD", "UD_ratio", "weighted_UD", "LD", "LD_ratio", "weighted_LD", "final_score", "matched", "changed"
        ])
        for result in results_round3:
            # 只有最终被选中的 candidate（matched==True）才判断 changed，其余为""
            if result["matched"]:
                changed_flag = (best2.get(result["pending"]) != result["candidate"])
            else:
                changed_flag = ""
            writer.writerow([
                result["pending"],
                result["candidate"],
                f"{result['prob']:.4f}" if result["prob"] is not None else "None",
                f"{result['UD']:.4f}",
                f"{result['UD_ratio']:.4f}",
                f"{result['weighted_UD']:.4f}",
                f"{result['LD']:.4f}",
                f"{result['LD_ratio']:.4f}",
                f"{result['weighted_LD']:.4f}",
                f"{result['final_score']:.4f}",
                result["matched"],
                changed_flag
            ])
    print(f"第三轮匹配结果已保存到 {output_file3}")
    if changed:
        print(f"第三轮有{len(changed)}个颗粒匹配发生修正，详情见输出文件。")

    # 保存无候选和锚点不足信息（只保存一次即可）
    with open(os.path.join(args.output_dir, 'pending_no_candidate.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["pending_label"])
        for p in no_candidate_pending:
            writer.writerow([p])
    with open(os.path.join(args.output_dir, 'pending_skipped_anchor_round2.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["pending", "anchor_count", "candidate_count"])
        writer.writeheader()
        for row in skipped_pending2:
            writer.writerow(row)
    with open(os.path.join(args.output_dir, 'pending_skipped_anchor_round3.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["pending", "anchor_count", "candidate_count"])
        writer.writeheader()
        for row in skipped_pending3:
            writer.writerow(row)

if __name__ == '__main__':
    main()
