# 该脚本用于统计不排除锚点时，匹配结果为锚点颗粒的pending颗粒数量

import csv
import argparse
import numpy as np

def read_best_match(csv_file):
    # 读取csv，返回pending_label到candidate_label的映射（只保留matched==True的行）
    mapping = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('matched', 'False') in ['True', 'TRUE', '1', True]:
                mapping[int(row['pending_label'])] = int(row['candidate_label'])
    return mapping

def read_anchor_ref_labels(init_match_file):
    # 读取初始匹配文件，返回所有锚点的ref_label集合
    data = np.load(init_match_file, allow_pickle=True)
    anchor_particles = data['anchor_particles'].item()
    anchor_ref_labels = set([info["predicted_reference_label"] for info in anchor_particles.values()])
    return anchor_ref_labels

def main():
    parser = argparse.ArgumentParser(description="统计不排除锚点时，匹配结果为锚点颗粒的pending颗粒数量")
    parser.add_argument('--without_anchor_csv', default='/share/home/202321008879/data/track_results/load1_aitoorigin_ai(old)/matchbyanchor/(不排除anchor作为候选)matchbyanchor_results_round3.csv', help="不排除锚点的round3 csv文件")
    parser.add_argument('--init_match_file', default='/share/home/202321008879/data/track_results/load1_aitoorigin_ai(old)/load1toorigin_initmatch.npz', help="初始匹配npz文件（用于获取锚点ref标签）")
    args = parser.parse_args()

    match_without_anchor = read_best_match(args.without_anchor_csv)
    anchor_ref_labels = read_anchor_ref_labels(args.init_match_file)

    changed_to_anchor = [pending for pending, cand in match_without_anchor.items() if cand in anchor_ref_labels]

    print(f"在不排除锚点时，有 {len(changed_to_anchor)} 个颗粒的匹配结果为锚点颗粒。")
    if changed_to_anchor:
        print("这些pending_label如下：")
        print(changed_to_anchor)

if __name__ == '__main__':
    main()