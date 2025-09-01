# 实现根据def标签查找ref体系内距离映射位置小于2d50的候选颗粒
# 但是实际查询出来比结果中的候选颗粒多，多了别的颗粒，不知道怎么回事
import argparse
import numpy as np
from scipy.spatial import KDTree
import h5py
import os

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

def get_prediction_for_particle_from_h5(h5_path, pending_particle):
    with h5py.File(h5_path, 'r') as h5f:
        particles_grp = h5f['particles']
        pid = str(pending_particle)
        if pid not in particles_grp:
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

def main():
    parser = argparse.ArgumentParser(description="根据def标签，查找ref体系内距离映射位置小于2d50的候选颗粒")
    parser.add_argument("--def_label", type=int, required=True, help="待查询的def标签")
    parser.add_argument("--ref_type", required=True, help="参考体系类型")
    parser.add_argument("--def_type", required=True, help="变形体系类型")
    parser.add_argument('--ref_npz', default=None, help="参考体系颗粒属性 npz 文件")
    parser.add_argument('--def_npz', default=None, help="变形体系颗粒属性 npz 文件")
    parser.add_argument('--d50', type=float, default=0.72, help="D50 值")
    parser.add_argument("--inter_strain", type=float, default=None, help="阶段间应变值（用于映射位置）")
    parser.add_argument('--pred_h5', default=None, help="预测结果 h5 文件（可选）")
    args = parser.parse_args()

    # 自动补全文件路径
    if args.ref_npz is None:
        args.ref_npz = f"/share/home/202321008879/data/sand_propertities/{args.ref_type}_ai_particle_properties/{args.ref_type}_particle_properties.npz"
    if args.def_npz is None:
        args.def_npz = f"/share/home/202321008879/data/sand_propertities/{args.def_type}_ai_particle_properties/{args.def_type}_particle_properties.npz"
    if args.pred_h5 is None:
        args.pred_h5 = f"/share/home/202321008879/data/track_results/{args.def_type}to{args.ref_type}/{args.def_type}_ai_pred.h5"

    # 应变映射
    strain_mapping = {"origin": 0.0, "load1": 0.01, "load5": 0.05, "load10": 0.1, "load15": 0.15, 'originnew_labbotm1k':0.0,'load1_selpar':0.01}
    if args.inter_strain is None:
        args.inter_strain = strain_mapping[args.def_type] - strain_mapping[args.ref_type]
    org_sample_height = 58.57788148034935

    # 加载属性
    ref_mapping = load_properties(args.ref_npz)
    def_mapping = load_properties(args.def_npz)

    if args.def_label not in def_mapping:
        print(f"def标签 {args.def_label} 不在def体系属性文件中")
        return

    # 获取def颗粒质心
    pend_centroid = def_mapping[args.def_label]['corrected_centroid(physical)']
    # 计算映射位置
    def_sample_height = np.max([v['corrected_centroid(physical)'][2] for v in def_mapping.values()])
    relative_z = pend_centroid[2] / def_sample_height
    displacement = np.array([0, 0, args.inter_strain * relative_z * org_sample_height])
    mapped_position = pend_centroid + displacement

    # 构建ref KDTree
    ref_labels = list(ref_mapping.keys())
    ref_centroids = np.array([ref_mapping[lab]['corrected_centroid(physical)'] for lab in ref_labels])
    ref_kdtree = KDTree(ref_centroids)

    # 查询2d50范围内的ref颗粒
    search_radius = 4 * args.d50
    idxs = ref_kdtree.query_ball_point(mapped_position, r=search_radius)

    print(f"def标签: {args.def_label}")
    print(f"映射位置: {mapped_position}")
    print(f"搜索半径: {search_radius}")
    print(f"候选ref颗粒数: {len(idxs)}")

    mapping = {}
    # 输出候选颗粒信息
    for idx in idxs:
        ref_lab = ref_labels[idx]
        ref_centroid = ref_centroids[idx]
        distance = np.linalg.norm(ref_centroid - mapped_position)/(args.d50)
        info = {
            "ref_label": ref_lab,
            "ref_centroid": ref_centroid.tolist(),
            "distanceD50": distance
        }
        mapping[ref_lab]=distance

        # 可选：读取预测概率
        if os.path.exists(args.pred_h5):
            prediction = get_prediction_for_particle_from_h5(args.pred_h5, args.def_label)
            if prediction is not None:
                ref_idx = ref_lab - 1
                ref_probs = []
                for subset in prediction["subsets"]:
                    prob_arr = subset["probability"]
                    if 0 <= ref_idx < len(prob_arr):
                        ref_probs.append(prob_arr[ref_idx])
                avg_prob = np.mean(ref_probs) if ref_probs else None
                info["avg_prob"] = avg_prob
    # print(info)
    lab = [833,1159,1570,1887,3732,5481,6683,7029,
           7350,8044,8196,10105,10578,10940,11096,11452,12404,12636,15646,16334,18393,19097,19616,21107,21829,22009,23883,24511,25837]
    for label in lab:
        if label in mapping:
            print(f"{label}, {mapping[label]:.2f}")
        else:
            print(f"ref_label: {label} 不在候选列表中")

if __name__ == '__main__':
    main()
