# 本来想着是用来模拟某一轮用来计算的锚点颗粒的，不过对比了发现不太一样，这个不用
import argparse
import numpy as np
import csv
from scipy.spatial import KDTree

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

def build_kdtree(anchor_data):
    anchor_labels = list(anchor_data.keys())
    anchor_def_centroids = np.array([anchor_data[label]["def_centroid"] for label in anchor_labels])
    kdtree = KDTree(anchor_def_centroids)
    return kdtree, anchor_labels

def get_nearby_anchors(pending_particle, kdtree, anchor_labels, def_mapping, anchor_radius, K=10, restrict_radius=None):
    pending_centroid = def_mapping[pending_particle]['corrected_centroid(physical)']
    if restrict_radius:
        idxs = kdtree.query_ball_point(pending_centroid, r=anchor_radius)
        if len(idxs) < K:
            return []
        dists = [np.linalg.norm(kdtree.data[i] - pending_centroid) for i in idxs]
        sorted_idxs = [idx for _, idx in sorted(zip(dists, idxs))]
        chosen = sorted_idxs[:K]
        return [anchor_labels[i] for i in chosen]
    else:
        dists, nearest = kdtree.query(pending_centroid, k=K+1)  # 多查一个以防A被包含
        if np.isscalar(nearest):
            nearest = [nearest]
        # 排除待匹配颗粒自身
        filtered_nearest = [i for i in nearest if anchor_labels[i] != pending_particle]
        return [anchor_labels[i] for i in filtered_nearest[:K]]  # 取前K个

def load_anchor_data_from_csv(csv_path):
    # 读取csv，返回pending->candidate的最佳匹配
    anchor_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('matched', 'False') == 'True':
                pending = int(row['pending_label'])
                candidate = int(row['candidate_label'])
                anchor_dict[pending] = candidate
    return anchor_dict

def main():
    parser = argparse.ArgumentParser(description="打印某颗粒计算UD/LD时所用锚点颗粒信息")
    parser.add_argument('--csv', default='/share/home/202321008879/data/track_results/comparison/load15toload10/half/matchbyanchor_half19.csv', 
                        help='某一轮匹配结果csv')
    parser.add_argument('--pending', type=int, default = 10001, help='待查询的pending颗粒标签')
    
    parser.add_argument('--def_type', default='load15', help='变形体系属性npz')
    parser.add_argument('--ref_type', default='load10', help='参考体系属性npz')
    parser.add_argument('--anchor_radius', type=float, default=2.88, help='锚点搜索半径（如4*d50）')
    parser.add_argument('--K', type=int, default=10, help='最近K个锚点')
    parser.add_argument('--restrict_radius', default=None, help='是否限制半径')
    args = parser.parse_args()


    ref_npz = f"/share/home/202321008879/data/sand_propertities/{args.ref_type}_ai_particle_properties/{args.ref_type}_particle_properties.npz"
    def_npz = f"/share/home/202321008879/data/sand_propertities/{args.def_type}_ai_particle_properties/{args.def_type}_particle_properties.npz"
    # 加载属性
    def_mapping = load_properties(def_npz)
    ref_mapping = load_properties(ref_npz)

    # 读取锚点（即已匹配的pending->candidate）
    anchor_dict = load_anchor_data_from_csv(args.csv)

    # 构建anchor_data
    anchor_data = {}
    for def_label, ref_label in anchor_dict.items():
        anchor_data[def_label] = {
            "ref_label": ref_label,
            "def_centroid": def_mapping[def_label]['corrected_centroid(physical)'].tolist(),
            "ref_centroid": ref_mapping[ref_label]['corrected_centroid(physical)'].tolist(),
        }

    # 构建KDTree
    kdtree, anchor_labels = build_kdtree(anchor_data)

    # 查询pending颗粒所用锚点
    used_anchors = get_nearby_anchors(
        pending_particle=args.pending,
        kdtree=kdtree,
        anchor_labels=anchor_labels,
        def_mapping=def_mapping,
        anchor_radius=args.anchor_radius,
        K=args.K,
        restrict_radius=args.restrict_radius
    )

    if not used_anchors:
        print(f"未找到足够的锚点（K={args.K}）")
        return

    print(f"用于pending颗粒 {args.pending} 计算的锚点颗粒（def_label, ref_label）：")
    for def_label in used_anchors:
        ref_label = anchor_data[def_label]["ref_label"]
        print(f"def_label: {def_label}, ref_label: {ref_label}")

if __name__ == '__main__':
    main()
