# 从final_results随机抽取100个def标签及其预测ref标签
import argparse
import numpy as np
import os
import csv

def load_final_results(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    results = data['results'].item()
    return results

def sample_labels(results, num_samples=100, seed=42):
    def_labels = list(results.keys())
    np.random.seed(seed)
    if len(def_labels) < num_samples:
        raise ValueError(f"可用def标签数不足{num_samples}，实际为{len(def_labels)}")
    sampled = np.random.choice(def_labels, size=num_samples, replace=False)
    return sampled

def save_sampled_csv(sampled_labels, results, output_csv):
    sorted_labels = sorted(sampled_labels)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['def_label', 'predicted_reference_label'])
        for def_lab in sorted_labels:
            ref_lab = results[def_lab]['predicted_reference_label']
            writer.writerow([def_lab, ref_lab])
    print(f"已保存抽样结果到: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="从final_results随机抽取100个def标签及其预测ref标签")
    parser.add_argument('--def_type', required=True, help='变形体系类型，例如 origin, load1, load5, load10, load15')
    parser.add_argument('--ref_type', required=True, help='参考体系类型')
    parser.add_argument('--final_results_npz', default=None, help='final_results.npz文件路径')
    parser.add_argument('--output_csv', default=None, help='输出csv路径')
    parser.add_argument('--num_samples', type=int, default=100, help='抽样数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    data_dir = '/share/home/202321008879/data/'
    if args.final_results_npz is None:
        args.final_results_npz = os.path.join(
            data_dir, 'track_results', f'{args.def_type}to{args.ref_type}', 'final_results.npz'
        )
    if args.output_csv is None:
        args.output_csv = os.path.join(
            data_dir, 'track_results', f'{args.def_type}to{args.ref_type}', f'sampled_{args.num_samples}_final_results.csv'
        )

    results = load_final_results(args.final_results_npz)
    sampled_labels = sample_labels(results, num_samples=args.num_samples, seed=args.seed)
    save_sampled_csv(sampled_labels, results, args.output_csv)

if __name__ == '__main__':
    main()
