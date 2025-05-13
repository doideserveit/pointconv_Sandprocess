# 该脚本用于检测颗粒周围以2倍d50为半径的区域内的颗粒个数，并计算局部布局差异
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import csv
from scipy.spatial import KDTree  # 导入 KDTree


# 获取候选颗粒的个数及标签，并计算预测概率
def get_candidate_particles(pending_particles, ref_mapping, def_mapping, d50, def_sample_height, predictions_file, strain, anchor_particles):
    """
    获取候选颗粒的个数及标签，并计算预测概率。
    pending_particles: 待匹配颗粒的标签集合
    ref_mapping: 参考体系颗粒属性映射
    def_mapping: 变形体系颗粒属性映射
    d50: D50 值
    def_sample_height: 变形体系样本高度
    predictions_file: 预测结果文件路径
    strain: 应变值
    anchor_particles: 锚点颗粒信息，用于排除锚点颗粒
    返回值: 包含候选颗粒信息和预测概率的字典
    """
    # 加载预测结果
    pred_data = np.load(predictions_file, allow_pickle=True)
    predictions = pred_data['results'].item()

    search_radius = 2 * d50
    candidate_info = {}

    for pending_particle in pending_particles:
        pend_centroid = def_mapping[pending_particle]['corrected_centroid(physical)']
        relative_z = pend_centroid[2] / def_sample_height
        displacement = np.array([0, 0, strain * relative_z * def_sample_height])
        mapped_position = pend_centroid + displacement

        candidate_info[pending_particle] = []
        for ref_lab, ref_info in ref_mapping.items():
            # 排除锚点颗粒
            if ref_lab in anchor_particles:
                continue

            ref_centroid = ref_info['corrected_centroid(physical)']
            distance = np.linalg.norm(ref_centroid - mapped_position)
            if distance <= search_radius:
                # 获取预测概率
                if pending_particle in predictions:
                    subsets = predictions[pending_particle]["subsets"]
                    ref_probs = [subset["probability"][ref_lab - 1] for subset in subsets]
                    avg_prob = np.mean(ref_probs)
                else:
                    avg_prob = None  # 如果没有预测结果，设置为 None

                # 保存候选颗粒信息
                candidate_info[pending_particle].append({
                    "candidate_label": ref_lab,
                    "distance": distance,
                    "avg_prob": avg_prob
                })

    return candidate_info

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

def get_nearby_anchors(pending_particle, kdtree, anchor_labels, def_mapping, radius, K):
    """
    混合策略：先以半径 radius 搜索锚点，不足 K 个时再补齐最近 K 个。
    返回锚点标签列表。
    """
    pending_centroid = def_mapping[pending_particle]['corrected_centroid(physical)']
    # 半径搜索
    idxs = kdtree.query_ball_point(pending_centroid, r=radius)
    if len(idxs) >= K:
        chosen = idxs
    else:
        # 先取 radius 内所有
        chosen = list(idxs)
        # 再取最近补齐
        dists, nearest = kdtree.query(pending_centroid, k=K)
        # nearest 可能是标量或列表
        if np.isscalar(nearest):
            nearest = [nearest]
        for idx in nearest:
            if idx not in chosen:
                chosen.append(idx)
                if len(chosen) >= K:
                    break
    return [anchor_labels[i] for i in chosen]


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


def compute_weighted_deviation_indicators(vals, weight):
    """
    根据公式计算 UD或LD 的比值和加权结果。
    vals: 候选颗粒的 UD 或 LD 值列表
    weight: UD或LD 的权重 
    返回值: 包含 UD/LD、比值和加权结果的列表
    """
    vmax = max(vals) if vals else 1  # 避免除以 0

    results = []
    for v in vals:
        r = v / vmax  # 计算比值
        results.append({
            "value": v,
            "ratio": r,
            "weighted": weight*r
        })
    return results


def process_candidates_final_score(pending_particles, candidate_info, kdtree, anchor_labels, anchor_data,
                                   def_mapping, ref_mapping, radius, K, wUD, wLD):
    """
    对每个待匹配颗粒：
    1. 统计半径 radius (2×d50) 和 2×radius (4×d50) 内锚点数
    2. 找邻域锚点（radius + 补齐 K 个）
    3. 用 compute_UD/compute_LD 计算 UD/LD
    4. 归一化 + 加权
    5. + (1 - avg_prob) 得到最终分数
    6. 选最小分数作为匹配
    返回结果列表，包含额外的统计
    """
    results = []
    for p in pending_particles:
        cands = candidate_info.get(p, [])
        if not cands:
            continue
        # 统计不同半径内锚点数
        pending_centroid = def_mapping[p]['corrected_centroid(physical)']
        idx_small = kdtree.query_ball_point(pending_centroid, r=radius)
        idx_large = kdtree.query_ball_point(pending_centroid, r=2 * radius)
        count_small = len(idx_small)
        count_large = len(idx_large)
        # 获取用于匹配的锚点（radius + 补齐 K 个）
        nearby = get_nearby_anchors(p, kdtree, anchor_labels, def_mapping, radius, K)
        labels = [c['candidate_label'] for c in cands]
        probs = [c['avg_prob'] for c in cands]
        # 计算误差
        ud = compute_UD(p, labels, nearby, anchor_data, def_mapping, ref_mapping)
        ld = compute_LD(p, labels, nearby, anchor_data, def_mapping, ref_mapping)
        # 归一化加权
        ud_vals = [ud[lab] for lab in labels]
        ld_vals = [ld[lab] for lab in labels]
        ud_info = compute_weighted_deviation_indicators(ud_vals, wUD)
        ld_info = compute_weighted_deviation_indicators(ld_vals, wLD)
        # 最终得分 = 加权 UD + 加权 LD + (1 - probs[i])
        scores = [ud_info[i]['weighted'] + ld_info[i]['weighted'] + (1 - probs[i])
                  for i in range(len(labels))]
        best = int(np.argmin(scores))
        # 保存
        for i, lab in enumerate(labels):
            results.append({
                'pending': p,
                'candidate': lab,
                'prob': probs[i],
                'count_2d50': count_small,
                'count_4d50': count_large,
                'UD': ud[lab],
                'UD_ratio': ud_info[i]['ratio'],
                'weighted_UD': ud_info[i]['weighted'],
                'LD': ld[lab],
                'LD_ratio': ld_info[i]['ratio'],
                'weighted_LD': ld_info[i]['weighted'],
                'final_score': scores[i],
                'matched': i == best
            })
    return results


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
    parser.add_argument('--ref_npz', default='/share/home/202321008879/data/sand_propertities/originnew_labbotm1k_particle_properties/originnew_labbotm1k_particle_properties.npz', 
                        help="参考体系颗粒属性 npz 文件")
    parser.add_argument('--def_npz', default='/share/home/202321008879/data/sand_propertities/Load1_selpar_particle_properties/Load1_selpar_particle_properties.npz', 
                        help="变形体系颗粒属性 npz 文件")
    parser.add_argument('--output_dir', default='/share/home/202321008879/data/track_results/load1_selpartooriginnew_labbotm1k/matchbyanchor', 
                        help="输出文件路径")
    parser.add_argument('--predictions_file', default='/share/home/202321008879/data/track_results/load1_selpartooriginnew_labbotm1k/predictions.npz', 
                        help="预测结果 npz 文件")
    parser.add_argument('--init_match_file', default='/share/home/202321008879/data/track_results/' \
                        'load1_selpartooriginnew_labbotm1k/load1_selpartooriginnew_labbotm1k_initmatch.npz',)
    parser.add_argument('--d50', type=float, default=0.72, help="D50 值")
    parser.add_argument("--strain", type=float, default=0.01, help="应变值")
    parser.add_argument("--radius", type=float, default=1.44, help="锚点搜索半径（默认2×d50）")
    parser.add_argument("--K", type=int, default=5, help="每个待匹配颗粒至少返回的锚点总数")

    args = parser.parse_args()

    # 加载初次匹配结果
    init_match_data = np.load(args.init_match_file, allow_pickle=True)
    anchor_particles = init_match_data['anchor_particles'].item()
    pending_particles = init_match_data['pending_particles']
    sample_height = init_match_data['sample_height'].item()
    def_sample_height = init_match_data['def_sample_height'].item()

    # 加载属性数据
    ref_mapping = load_properties(args.ref_npz)
    def_mapping = load_properties(args.def_npz)
    
    # 构建锚点数据为字典形式
    anchor_data = {}
    for def_label, anchor_info in anchor_particles.items():
        ref_label = anchor_info["predicted_reference_label"]
        anchor_data[def_label] = {
            "ref_label": ref_label,
            "def_centroid": def_mapping[def_label]['corrected_centroid(physical)'].tolist(),
            "ref_centroid": ref_mapping[ref_label]['corrected_centroid(physical)'].tolist(),
            "displacement": anchor_info["displacement(mm)"],
        }

    # 构建 KD-Tree
    kdtree, anchor_labels = build_kdtree(anchor_data)

    # 计算 UD 和 LD 的权重
    weightUD = args.strain / (args.strain + 0.05)  
    weightLD = 1 - weightUD

    # 构建候选颗粒信息，排除锚点颗粒
    candidate_info = get_candidate_particles(
        pending_particles=pending_particles,
        ref_mapping=ref_mapping,
        def_mapping=def_mapping,
        d50=args.d50,
        def_sample_height=def_sample_height,
        predictions_file=args.predictions_file,
        strain=args.strain,
        anchor_particles=anchor_particles  # 传入锚点颗粒信息
    )

    # 处理候选颗粒并计算最终分数
    results = process_candidates_final_score(
        pending_particles=pending_particles,
        candidate_info=candidate_info,
        kdtree=kdtree,
        anchor_labels=anchor_labels,
        anchor_data=anchor_data,
        def_mapping=def_mapping,
        ref_mapping=ref_mapping,
        radius=args.radius,
        K=args.K,
        wUD=weightUD,
        wLD=weightLD
    )

    # 保存结果到 CSV 文件
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'matchbyanchor_results.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "pending_label", "candidate_label", "avg_prob", "count_2d50", "count_4d50",
            "UD", "UD_ratio", "weighted_UD", "LD", "LD_ratio", "weighted_LD", "final_score", "matched"
        ])
        for result in results:
            writer.writerow([
                result["pending"],
                result["candidate"],
                f"{result['prob']:.4f}" if result["prob"] is not None else "None",
                result["count_2d50"],
                result["count_4d50"],
                f"{result['UD']:.4f}",
                f"{result['UD_ratio']:.4f}",
                f"{result['weighted_UD']:.4f}",
                f"{result['LD']:.4f}",
                f"{result['LD_ratio']:.4f}",
                f"{result['weighted_LD']:.4f}",
                f"{result['final_score']:.4f}",
                result["matched"]
            ])
    print(f"匹配结果已保存到 {output_file}")

if __name__ == '__main__':
    main()
