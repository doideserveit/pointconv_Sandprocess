# 该脚本用于计算所有anchor颗粒的映射点与预测reference颗粒的距离，并计算该距离为多少倍d50。它会输出一个CSV文件和一个直方图。
import argparse
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def load_properties(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels = data['label']
    centroids = data['corrected_centroid(physical)']
    mapping = {}
    for i, lab in enumerate(labels):
        mapping[lab] = np.array(centroids[i])
    return mapping

def main():
    parser = argparse.ArgumentParser(description="计算所有anchor颗粒的映射点与预测reference颗粒的距离及其为多少倍d50")
    parser.add_argument('--def_type', type=str, default=15, help='属性文件路径')
    parser.add_argument('--init_match_file', type=str, 
                        default='/share/home/202321008879/data/track_results/load15toload10/load15toload10_initmatch_4d50.npz', 
                        help='初始匹配结果文件路径')
    parser.add_argument('--output_csv', type=str, 
                        default='/share/home/202321008879/data/track_results/load15toload10/candidate_distances_4d50.csv', help='输出csv文件名')
    parser.add_argument('--output_hist', type=str, 
                        default='/share/home/202321008879/data/track_results/load15toload10/candidate_distances_hist_4d50.png', help='输出直方图文件名')
    args = parser.parse_args()
    d50 = 0.72

    name = ['origin', 'load1', 'load5', 'load10', 'load15']
    type = [0, 1, 5, 10, 15]
    i = type.index(int(args.def_type))
    def_type = name[i]
    ref_type = name[i - 1]
    strain = type[i]*0.01 - type[i-1]*0.01
    property_dir = '/share/home/202321008879/data/sand_propertities'
    
    def_npz = os.path.join(property_dir, f'{def_type}_ai_particle_properties/{def_type}_particle_properties.npz')
    ref_npz = os.path.join(property_dir, f'{ref_type}_ai_particle_properties/{ref_type}_particle_properties.npz') 
    init_match_data = np.load(args.init_match_file, allow_pickle=True)
    def_sample_height = init_match_data['def_sample_height'].item() 
    org_sample_height = 58.57788148034935

    def_mapping = load_properties(def_npz)
    ref_mapping = load_properties(ref_npz)

    anchor_particles = init_match_data['anchor_particles'].item()

    results = []
    multiples = []

    for anchor_label, info in anchor_particles.items():
        pred_label = info.get('predicted_reference_label')
        if anchor_label not in def_mapping or pred_label not in ref_mapping:
            continue
        centroid1 = def_mapping[anchor_label]
        centroid2 = ref_mapping[pred_label]
        relative_z = centroid1[2] / def_sample_height
        displacement = np.array([0, 0, strain * relative_z * org_sample_height])
        mapped_position = centroid1 + displacement
        distance = np.linalg.norm(centroid2 - mapped_position)
        multiple = distance / d50
        results.append([anchor_label, pred_label, *mapped_position, *centroid2, distance, multiple])
        multiples.append(multiple)

    # 写入csv
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['anchor_label', 'predicted_label', 'mapped_x', 'mapped_y', 'mapped_z', 'pred_x', 'pred_y', 'pred_z', 'distance', 'd50_multiple'])
        writer.writerows(results)

    # 绘制直方图
    plt.figure(figsize=(8, 5))
    plt.hist(multiples, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Distance / d50')
    plt.ylabel('Count')
    plt.title('Distribution of Distance (in d50 multiples)')
    plt.tight_layout()
    plt.savefig(args.output_hist)
    plt.close()

    print(f"已输出csv到 {args.output_csv}，直方图到 {args.output_hist}")

if __name__ == '__main__':
    main()
