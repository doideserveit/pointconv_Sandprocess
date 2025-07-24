import os
import numpy as np
import csv
import argparse

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

def merge_results_with_displacement(init_match_data, matchbyanchor_results, def_mapping, ref_mapping):
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

    for result in matchbyanchor_results:
        def_lab = result["pending"]
        if def_lab not in merged_results:
            # 对于 pending 颗粒，计算其位移
            candidate_label = result["candidate"]
            displacement = calculate_displacement(def_mapping, ref_mapping, def_lab, candidate_label)
            merged_results[def_lab] = {
                "predicted_reference_label": candidate_label,
                "avg_prediction": result["prob"],
                "type": "pending",
                "displacement": displacement
            }
        else:
            print(f"Warning: 重复的 def_lab {def_lab}，已在 Init_match 中匹配，仍出现在 matchbyanchor 结果中。")

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

def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="Merge and process particle tracking results.")
    parser.add_argument('--ref_type', required=True, help='参考体系类型，例如 origin, load1, load5, load10, load15')
    parser.add_argument('--def_type', required=True, help='变形体系类型，例如 origin, load1, load5, load10, load15')

    parser.add_argument('--init_match_file', default=None, help='Path to the init match results file.')
    parser.add_argument('--matchbyanchor_file', default=None, help='Path to the match by anchor results file.')
    parser.add_argument('--def_properties_file', default=None, help='Path to the deformed properties file.')
    parser.add_argument('--ref_properties_file', default=None, help='Path to the reference properties file.')
    parser.add_argument('--output_dir', default=None, help='Output file path.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data_dir = '/share/home/202321008879/data/'
    # 自动补全文件路径
    if args.init_match_file is None:
        args.init_match_file = os.path.join(data_dir, 'track_results', f'{args.def_type}to{args.ref_type}', f'{args.def_type}to{args.ref_type}_initmatch.npz')
    if args.matchbyanchor_file is None:
        args.matchbyanchor_file = os.path.join(data_dir, 'track_results', f'{args.def_type}to{args.ref_type}', 'matchbyanchor', 'matchbyanchor10.csv')
    if args.def_properties_file is None:
        args.def_properties_file = os.path.join(data_dir, 'sand_propertities', f'{args.def_type}_ai_particle_properties', f'{args.def_type}_particle_properties.npz')
    if args.ref_properties_file is None:
        args.ref_properties_file = os.path.join(data_dir, 'sand_propertities', f'{args.ref_type}_ai_particle_properties', f'{args.ref_type}_particle_properties.npz')
    if args.output_dir is None:
        args.output_dir = os.path.join(data_dir, 'track_results', f'{args.def_type}to{args.ref_type}')
    output_csv_file = os.path.join(args.output_dir, 'final_results.csv')
    output_npz_file = os.path.join(args.output_dir, 'final_results.npz')


    # 加载和合并结果
    init_match_data = load_init_match_results(args.init_match_file)
    matchbyanchor_results = load_matchbyanchor_results(args.matchbyanchor_file)
    def_mapping = load_properties(args.def_properties_file)
    ref_mapping = load_properties(args.ref_properties_file)
    merged_results = merge_results_with_displacement(init_match_data, matchbyanchor_results, def_mapping, ref_mapping)

    # 保存结果
    save_results(merged_results, output_csv_file, output_npz_file)


