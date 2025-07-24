import os
import numpy as np
import csv

def load_init_match_results(init_match_file):
    """
    加载 init_match 的结果。
    """
    data = np.load(init_match_file, allow_pickle=True)
    return data['anchor_particles'].item()

def load_matchbyanchor_results(matchbyanchor_file):
    """
    从 matchbyanchor 的 CSV 文件中读取匹配成功的结果。
    """
    results = []
    with open(matchbyanchor_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["matched"] == "True":  # 只读取匹配成功的结果
                results.append({
                    "pending": int(row["pending_label"]),
                    "candidate": int(row["candidate_label"]),
                    "prob": float(row["avg_prob"])
                })
    return results

def load_properties(properties_file):
    """
    加载颗粒属性文件，并构建映射。
    """
    data = np.load(properties_file, allow_pickle=True)
    labels = data['label']
    centroids_physical = data['corrected_centroid(physical)']
    mapping = {}
    for i, label in enumerate(labels):
        mapping[label] = {
            'corrected_centroid(physical)': np.array(centroids_physical[i])
        }
    return mapping

def calculate_displacement(def_mapping, ref_mapping, pending_label, candidate_label):
    """
    计算 pending 颗粒的位移。
    def_mapping: 变形体系颗粒属性映射
    ref_mapping: 参考体系颗粒属性映射
    pending_label: 待匹配颗粒标签
    candidate_label: 匹配的候选颗粒标签
    返回值: 位移标量
    """
    def_centroid = def_mapping[pending_label]['corrected_centroid(physical)']
    ref_centroid = ref_mapping[candidate_label]['corrected_centroid(physical)']
    displacement = np.linalg.norm(def_centroid - ref_centroid)
    return displacement

def merge_results_with_displacement(init_match_data):
    """
    合并 init_match 和 matchbyanchor 的结果，并计算位移参数。
    """
    merged_results = {}
    for def_lab, anchor_info in init_match_data.items():
        # 对于锚点颗粒，直接读取位移值
        displacement = anchor_info["displacement(mm)"]
        merged_results[def_lab] = {
            "predicted_reference_label": anchor_info["predicted_reference_label"],
            "avg_prediction": anchor_info["avg_prediction"],
            "type": "anchor",
            "displacement": displacement
        }


    return dict(sorted(merged_results.items(), key=lambda x: x[0]))

def save_results(sorted_results, output_csv_file, output_npz_file):
    """
    保存结果到 CSV 和 NPZ 文件。
    """
    # 保存到 CSV 文件
    with open(output_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["def_lab", "predicted_reference_label", "avg_prediction", "type", "displacement"])
        for def_lab, info in sorted_results.items():
            writer.writerow([
                def_lab,
                info["predicted_reference_label"],
                f"{info['avg_prediction']:.4f}",
                info["type"],
                f"{info['displacement']:.4f}"
            ])
    print(f"结果已保存到 CSV 文件: {output_csv_file}")

    # 保存到 NPZ 文件
    np.savez(output_npz_file, results=sorted_results)
    print(f"结果已保存到 NPZ 文件: {output_npz_file}")

if __name__ == "__main__":
    list1 = ['/share/home/202321008879/data/track_results/load5toorigin/load5toorigin_initmatch.npz',
            '/share/home/202321008879/data/track_results/load10toorigin/load10toorigin_initmatch.npz'
            ]
    list2 = ['/share/home/202321008879/data/sand_propertities/load5_ai_particle_properties/load5_particle_properties.npz',
            '/share/home/202321008879/data/sand_propertities/load10_ai_particle_properties/load10_particle_properties.npz',
            ]
    list3 = ['/share/home/202321008879/data/track_results/load5toorigin',
            '/share/home/202321008879/data/track_results/load10toorigin',
            ]
    for i in range(len(list1)):
        # 文件路径
        init_match_file = list1[i]
        matchbyanchor_file = '/share/home/202321008879/data/track_results/load1toorigin/matchbyanchor/matchbyanchor_results_round3.csv'
        def_properties_file = list2[i]
        ref_properties_file = '/share/home/202321008879/data/sand_propertities/origin_ai_particle_properties/origin_particle_properties.npz'
        out_put_dir = list3[i]
        output_csv_file = os.path.join(out_put_dir, 'final_results.csv')
        output_npz_file = os.path.join(out_put_dir, 'final_results.npz')

        # 加载和合并结果
        init_match_data = load_init_match_results(init_match_file)

        def_mapping = load_properties(def_properties_file)
        ref_mapping = load_properties(ref_properties_file)
        merged_results = merge_results_with_displacement(init_match_data)

        # 保存结果
        save_results(merged_results, output_csv_file, output_npz_file)


