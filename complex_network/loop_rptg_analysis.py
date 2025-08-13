# 这个脚本是按照环的大小分析各阶段Loop颗粒的RPTG分布情况，横轴为rptg组，结果没有可以参考的地方
# 废弃
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_loop_labels(npz_file, loop_sizes):
    """读取npz文件，返回每种环类型对应的颗粒label集合（去重）"""
    data = np.load(npz_file, allow_pickle=True)
    loop_labels = {}
    for size in loop_sizes:
        key = f"{size}_labels"
        if key in data:
            labels = data[key]
            # npz里可能是int或str，统一转int
            labels = [int(l) for l in labels]
            loop_labels[size] = labels
    return loop_labels

def read_rptg(rptg_file):
    """读取RPTG文件，返回lab到rptg值的映射"""
    df = pd.read_csv(rptg_file)
    return dict(zip(df['def_lab'], df['RPTG']))

def group_rptg_by_bins(labels, rptg_dict, bins):
    """统计labels对应的rptg分布，返回bin区间到数量的映射"""
    bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    counts = {b: 0 for b in bin_labels}
    for lab in labels:
        v = rptg_dict.get(lab)
        if v is None:
            continue
        try:
            v = float(v)
        except:
            continue
        assigned = False
        for i in range(len(bins)-2):
            if bins[i] <= v < bins[i+1]:
                counts[bin_labels[i]] += 1
                assigned = True
                break
        if not assigned and bins[-2] <= v <= bins[-1]:
            counts[bin_labels[-1]] += 1
    return counts

def save_csv(loop_rptg_counts, output_csv):
    """保存统计结果到csv"""
    rows = []
    for loop_size, bin_counts in loop_rptg_counts.items():
        total = sum(bin_counts.values())
        for bin_label, count in bin_counts.items():
            freq = count / total if total > 0 else 0
            rows.append({'loop_size': loop_size, 'rptg_bin': bin_label, 'count': count, 'freq': freq})
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"统计结果已保存到: {output_csv}")

def plot_loop_rptg(loop_rptg_counts, output_png):
    """绘制折线图"""
    plt.figure(figsize=(8,6))
    bins = None
    for loop_size, bin_counts in loop_rptg_counts.items():
        bins = list(bin_counts.keys())
        total = sum(bin_counts.values())
        if total == 0:
            continue
        freqs = [bin_counts[b]/total for b in bins]
        plt.plot(bins, freqs, marker='o', label=f"{loop_size}-loop (n={total})")
    plt.xlabel("RPTG")
    plt.ylabel("Frequency")
    plt.title("RPTG Distribution in Loops")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"折线图已保存到: {output_png}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="分析各阶段各组Loop颗粒的RPTG分布")
    parser.add_argument('--stage', type=str, required=True, help='阶段名，如load15')
    parser.add_argument('--loop_sizes', type=int, nargs='+', default=[3,4,5,6], help='分析的环大小')
    parser.add_argument('--bins', type=float, nargs='+', default=[0,0.1,0.2,0.3,0.4,0.5,0.6], help='RPTG分组区间')
    args = parser.parse_args()

    data_dir = '/share/home/202321008879/data'
    stage_list = ['origin', 'load1', 'load5', 'load10', 'load15']
    npz_file = os.path.join(data_dir, 'contact_network', args.stage, 'cycles.npz')
    prv_stage = stage_list[stage_list.index(args.stage) - 1 ]
    rptg_file = os.path.join(data_dir, 'rptg', f'{args.stage}to{prv_stage}_RPTG.csv')
    output_dir = os.path.join(data_dir, 'analysis', 'loop_rptg')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f'{args.stage}_loop_rptg.csv')
    output_png = os.path.join(output_dir, f'{args.stage}_loop_rptg.png')

    print(f"读取环文件: {npz_file}")
    print(f"读取RPTG文件: {rptg_file}")

    loop_labels = read_loop_labels(npz_file, args.loop_sizes)
    rptg_dict = read_rptg(rptg_file)
    bins = args.bins

    loop_rptg_counts = {}
    for size, labels in loop_labels.items():
        bin_counts = group_rptg_by_bins(labels, rptg_dict, bins)
        loop_rptg_counts[size] = bin_counts

    save_csv(loop_rptg_counts, output_csv)
    plot_loop_rptg(loop_rptg_counts, output_png)

if __name__ == '__main__':
    main()
