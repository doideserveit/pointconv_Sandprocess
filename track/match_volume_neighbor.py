import numpy as np
import argparse
import os
import csv
from scipy.spatial import KDTree

def load_properties(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels = data['label']
    centroids = data['corrected_centroid(physical)']
    volumes = data['physical_volume(mm)']
    mapping = {}
    for i, lab in enumerate(labels):
        mapping[lab] = {
            'centroid': np.array(centroids[i]),
            'volume': volumes[i]
        }
    return mapping

def get_mapped_position(def_centroid, strain, def_sample_height, org_sample_height):
    relative_z = def_centroid[2] / def_sample_height
    displacement = np.array([0, 0, strain * org_sample_height * relative_z])
    return def_centroid + displacement

def search_candidates(def_lab, def_mapping, ref_kdtree, ref_labels, ref_centroids, search_radius, strain, def_sample_height, org_sample_height):
    def_centroid = def_mapping[def_lab]['centroid']
    mapped_pos = get_mapped_position(def_centroid, strain, def_sample_height, org_sample_height)
    idxs = ref_kdtree.query_ball_point(mapped_pos, r=search_radius)
    return [ref_labels[i] for i in idxs]

def compute_volume_error(def_vol, ref_vol):
    if def_vol == 0:
        return np.inf
    return abs(def_vol - ref_vol) / def_vol

def get_neighbor_displacements(def_lab, matched, def_mapping, search_radius):
    def_centroid = def_mapping[def_lab]['centroid']
    matched_labels = list(matched.keys())
    if not matched_labels:
        return []
    matched_centroids = np.array([def_mapping[l]['centroid'] for l in matched_labels])
    kdtree = KDTree(matched_centroids)
    idxs = kdtree.query_ball_point(def_centroid, r=search_radius)
    neighbors = [matched_labels[i] for i in idxs if matched_labels[i] != def_lab]
    disps = [matched[n]['displacement'] for n in neighbors]
    return disps


def greedy_conflict_resolve_matching(
    def_mapping, ref_mapping, ref_kdtree, ref_labels, ref_centroids, search_radius,
    strain, def_sample_height, org_sample_height, vol_err_thresh, matched_prev=None, use_ud=False
):
    """
    按用户描述的贪心冲突解决流程进行匹配。
    """
    def_labels = list(def_mapping.keys())
    def_to_ref = {}
    ref_to_def = {}
    # 预先为每个def颗粒准备候选列表（按得分升序）
    candidate_lists = {}
    for def_lab in def_labels:
        candidates = search_candidates(def_lab, def_mapping, ref_kdtree, ref_labels, ref_centroids,
                                      search_radius, strain, def_sample_height, org_sample_height)
        def_vol = def_mapping[def_lab]['volume']
        def_centroid = def_mapping[def_lab]['centroid']
        cand_scores = []
        for ref_lab in candidates:
            ref_vol = ref_mapping[ref_lab]['volume']
            vol_err = compute_volume_error(def_vol, ref_vol)
            if vol_err >= vol_err_thresh:
                continue
            if use_ud and matched_prev is not None:
                neighbor_disps = get_neighbor_displacements(def_lab, matched_prev, def_mapping, search_radius)
                if len(neighbor_disps) == 0:
                    continue
                median_disp = np.median(np.stack(neighbor_disps), axis=0)
                disp = def_centroid - ref_mapping[ref_lab]['centroid']
                ud = np.linalg.norm(disp - median_disp) / (np.linalg.norm(median_disp) + 1e-6)
                score = vol_err + ud
            else:
                score = vol_err
            cand_scores.append((ref_lab, score, vol_err))
        # 按得分升序排序
        candidate_lists[def_lab] = sorted(cand_scores, key=lambda x: x[1])

    # 记录每个def当前尝试到第几个候选
    cand_idx = {def_lab: 0 for def_lab in def_labels}
    # 记录已丢失的def
    lost_defs = set()

    for def_lab in def_labels:
        current_def = def_lab
        while True:
            cands = candidate_lists[current_def]
            idx = cand_idx[current_def]
            if idx >= len(cands):
                # 没有可用候选，丢失
                lost_defs.add(current_def)
                break
            ref_lab, score, vol_err = cands[idx]
            # 检查ref是否已被占用
            if ref_lab not in ref_to_def:
                # 未被占用，直接匹配
                def_to_ref[current_def] = (ref_lab, score, vol_err)
                ref_to_def[ref_lab] = (current_def, score, vol_err)
                break
            else:
                # 已被占用，比较得分
                prev_def, prev_score, prev_vol_err = ref_to_def[ref_lab]
                if score < prev_score:
                    # 当前def更优，替换
                    def_to_ref[current_def] = (ref_lab, score, vol_err)
                    ref_to_def[ref_lab] = (current_def, score, vol_err)
                    # 被淘汰的def需要重新匹配，剔除该ref
                    cand_idx[prev_def] += 1
                    # 当前def匹配成功，继续处理被淘汰的def
                    current_def = prev_def
                    continue
                else:
                    # 当前def不是最优，尝试下一个候选
                    cand_idx[current_def] += 1
                    continue
    # 返回所有成功匹配的def
    final_matches = {def_lab: (ref_lab, score, vol_err) for def_lab, (ref_lab, score, vol_err) in def_to_ref.items()}
    return final_matches

def iterative_match(def_mapping, ref_mapping, ref_kdtree, ref_labels, ref_centroids, search_radius, strain, def_sample_height, org_sample_height, vol_err_thresh, max_rounds):
    print("第一轮匹配（体积误差）...")
    matched = {}
    first_round_matches = greedy_conflict_resolve_matching(
        def_mapping, ref_mapping, ref_kdtree, ref_labels, ref_centroids, search_radius,
        strain, def_sample_height, org_sample_height, vol_err_thresh, matched_prev=None, use_ud=False
    )
    for def_lab, (ref_lab, score, vol_err) in first_round_matches.items():
        disp = def_mapping[def_lab]['centroid'] - ref_mapping[ref_lab]['centroid']
        matched[def_lab] = {
            'ref_label': ref_lab,
            'volume_error': vol_err,
            'displacement': disp
        }
    print(f"第一轮最终匹配数: {len(matched)}")
    for round_idx in range(2, max_rounds+1):
        print(f"第{round_idx}轮匹配（体积+邻域位移）...")
        matches = greedy_conflict_resolve_matching(
            def_mapping, ref_mapping, ref_kdtree, ref_labels, ref_centroids, search_radius,
            strain, def_sample_height, org_sample_height, vol_err_thresh, matched_prev=matched, use_ud=True
        )
        new_matched = {}
        for def_lab, (ref_lab, score, vol_err) in matches.items():
            disp = def_mapping[def_lab]['centroid'] - ref_mapping[ref_lab]['centroid']
            new_matched[def_lab] = {
                'ref_label': ref_lab,
                'volume_error': vol_err,
                'ud_pct': score - vol_err,
                'displacement': disp
            }
        matched = new_matched
        print(f"第{round_idx}轮最终匹配数: {len(matched)}")
    return matched

def main():
    parser = argparse.ArgumentParser(description="基于体积和邻域位移的颗粒匹配算法")
    parser.add_argument("--ref_type", required=True, help="参考体系类型")
    parser.add_argument("--def_type", required=True, help="变形体系类型")
    parser.add_argument("--output_dir", default=None, help="输出目录")
    parser.add_argument("--d50", type=float, default=0.72, help="D50")
    parser.add_argument("--multiple", type=float, default=2, help="搜索半径倍数")
    parser.add_argument("--strain", type=float, default=0.01, help="阶段间应变")
    parser.add_argument("--org_sample_height", type=float, default=58.57788148034935, help="原始样本高度")
    parser.add_argument("--max_rounds", type=int, default=5, help="最大迭代轮数")
    parser.add_argument("--vol_err_thresh", type=float, default=0.10, help="体积误差阈值")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"/share/home/202321008879/data/track_results/vol_track/{args.def_type}to{args.ref_type}"
    os.makedirs(args.output_dir, exist_ok=True)
    search_radius = args.multiple * args.d50

    # 加载属性
    data_root = '/share/home/202321008879/data/'
    args.ref_npz = os.path.join(data_root, 'sand_propertities', f'{args.ref_type}_ai_particle_properties', f'{args.ref_type}_particle_properties.npz')
    args.def_npz = os.path.join(data_root, 'sand_propertities', f'{args.def_type}_ai_particle_properties', f'{args.def_type}_particle_properties.npz')
    ref_mapping = load_properties(args.ref_npz)
    def_mapping = load_properties(args.def_npz)
    ref_labels = list(ref_mapping.keys())
    ref_centroids = np.array([ref_mapping[lab]['centroid'] for lab in ref_labels])
    ref_kdtree = KDTree(ref_centroids)
    def_sample_height = max([def_mapping[lab]['centroid'][2] for lab in def_mapping.keys()])
    print(f"Def sample height (max z): {def_sample_height}")
    # ========== 第一轮+后续迭代 ==========

    matched = iterative_match(
        def_mapping, ref_mapping, ref_kdtree, ref_labels, ref_centroids, search_radius,
        args.strain, def_sample_height, args.org_sample_height, args.vol_err_thresh, args.max_rounds
    )
    # ========== 输出 ==========
    output_file = os.path.join(args.output_dir, "match_volume_neighbor_results.csv")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['def_label', 'ref_label', 'volume_error', 'ud_pct', 'displacement_x', 'displacement_y', 'displacement_z'])
        for def_lab, info in matched.items():
            row = [
                def_lab,
                info['ref_label'],
                info['volume_error'],
                info.get('ud_pct', ''),
                info['displacement'][0],
                info['displacement'][1],
                info['displacement'][2]
            ]
            writer.writerow(row)
    print(f"匹配结果已保存到 {output_file}")

if __name__ == '__main__':
    main()