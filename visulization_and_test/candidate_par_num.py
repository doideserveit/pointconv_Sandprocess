# 该脚本用于检测颗粒周围以2倍d50为半径的区域内的颗粒个数

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import csv

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

def analyze_candidate_predictions(candidate_labels, predictions_file, output_file):
    # 加载预测结果
    pred_data = np.load(predictions_file, allow_pickle=True)
    predictions = pred_data['results'].item()

    candidate_analysis = []
    for def_lab, ref_labels in candidate_labels.items():
        if def_lab not in predictions:
            continue

        # 获取 def_lab 的子集预测结果
        subsets = predictions[def_lab]["subsets"]

        for ref_lab in ref_labels:
            # 提取每个子集中 ref_lab 的预测概率
            ref_probs = [subset["probability"][ref_lab - 1] for subset in subsets]

            # 计算 ref_lab 的平均概率
            avg_prob = np.mean(ref_probs)

            # 保存待匹配颗粒和候选颗粒的预测概率
            candidate_analysis.append({
                "deformed_label": def_lab,
                "candidate_label": ref_lab,
                "avg_prob": avg_prob,
            })

    # 保存分析结果到文件
    analysis_output_file = os.path.splitext(output_file)[0] + '_analysis.csv'
    with open(analysis_output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["deformed_label", "candidate_label", "avg_prob"])
        for info in candidate_analysis:
            writer.writerow([
                info["deformed_label"],
                info["candidate_label"],
                f"{info['avg_prob']:.4f}",
            ])
    print(f"候选颗粒分析结果已保存到 {analysis_output_file}")

def main():
    parser = argparse.ArgumentParser(description="检测颗粒周围以2倍d50为半径的区域内的颗粒个数")
    parser.add_argument('--ref_npz', default='/share/home/202321008879/data/sand_propertities/originnew_labbotm1k_particle_properties/originnew_labbotm1k_particle_properties.npz', 
                        # /share/home/202321008879/data/sand_propertities/origin_ai/origin_particle_properties.npz
                        help="参考体系颗粒属性 npz 文件")
    parser.add_argument('--def_npz', default='/share/home/202321008879/data/sand_propertities/Load1_selpar_particle_properties/Load1_selpar_particle_properties.npz', 
                        # /share/home/202321008879/data/sand_propertities/load1_ai_particle_properties/load1_particle_properties.npz
                        help="变形体系颗粒属性 npz 文件")
    parser.add_argument('--output_file', default='/share/home/202321008879/project/sandprocess/visulization_and_test/selpar_candidate_par_num_results.csv', 
                        help="输出文件路径")
    parser.add_argument('--d50', type=float, default=0.72, help="D50 值")
    parser.add_argument("--strain", type=float, default=0.01, help="应变值")
    parser.add_argument('--predictions_file', default='/share/home/202321008879/data/track_results/load1_selpartooriginnew_labbotm1k/predictions.npz', 
                        # /share/home/202321008879/data/track_results/load1_aitoorigin_ai/load1_ai_pred.npz
                        help="预测结果 npz 文件")
    args = parser.parse_args()

    # 加载属性数据
    ref_mapping = load_properties(args.ref_npz)
    def_mapping = load_properties(args.def_npz)
    def_sample_height = max([def_mapping[lab]['corrected_centroid(physical)'][2] for lab in def_mapping.keys()])
    
    search_radius = 2 * args.d50
    candidate_counts = {}
    candidate_labels = {}

    for def_lab, def_info in def_mapping.items():
        def_centroid = def_info['corrected_centroid(physical)']
        relative_z = def_centroid[2] / def_sample_height
        displacement = np.array([0, 0, args.strain * relative_z * def_sample_height])
        mapped_position = def_centroid + displacement

        candidate_count = 0
        candidate_labels[def_lab] = []
        for ref_lab, ref_info in ref_mapping.items():
            ref_centroid = ref_info['corrected_centroid(physical)']
            distance = np.linalg.norm(ref_centroid - mapped_position)
            if distance <= search_radius:
                candidate_count += 1
                candidate_labels[def_lab].append(ref_lab)

        candidate_counts[def_lab] = candidate_count
        
    # 保存结果到文件
    with open(args.output_file, 'w') as f:
        f.write("deformed_label,candidate_count\n")
        for def_lab, count in candidate_counts.items():
            f.write(f"{def_lab},{count}\n")
    print(f"结果已保存到 {args.output_file}")

    # 绘制直方图
    plot_histogram(candidate_counts, args.output_file)

    # 分析候选颗粒的预测结果
    analyze_candidate_predictions(candidate_labels, args.predictions_file, args.output_file)

if __name__ == '__main__':
    main()
