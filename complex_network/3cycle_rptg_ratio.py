# 这个脚本计算了3环的颗粒中各RPTG组的比例，但是没意义。因为肯定是低RPTG组的颗粒多
# 意义与plot_cycle_group_trend.py类似，那个是按环颗粒计算，但是其分母是在所有环中出现的次数
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_stage_rptg_dict(stage, rptg_dir, final_results_file, prev_stage=None):
    """
    获取某阶段颗粒的rptg值映射
    origin: 读取load1toorigin_RPTG.csv和final_results.npz，反向映射ref_lab->def_lab，取均值
    其他: 读取{stage}to{prev_stage}_RPTG.csv，grain列为颗粒，rptg列为分组
    返回: {grain_id: rptg值(float)}
    """
    if stage == 'origin':
        rptg_csv = os.path.join(rptg_dir, 'load1toorigin_RPTG.csv')
        rptg_df = pd.read_csv(rptg_csv)
        rptg_map = dict(zip(rptg_df['def_lab'], rptg_df['RPTG']))
        data = np.load(final_results_file, allow_pickle=True)
        results = data['results'].item()
        ref2defs = {}
        for def_lab, info in results.items():
            ref_lab = int(info['predicted_reference_label'])
            ref2defs.setdefault(ref_lab, []).append(def_lab)
        ref_rptg = {}
        for ref_lab, def_labs in ref2defs.items():
            vals = []
            for d in def_labs:
                v = rptg_map.get(d)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except:
                        continue
            if vals:
                ref_rptg[ref_lab] = np.mean(vals)
        return ref_rptg
    else:
        # 非origin阶段，文件名为 {stage}to{prev_stage}_RPTG.csv
        assert prev_stage is not None
        csv_path = os.path.join(rptg_dir, f'{stage}to{prev_stage}_RPTG.csv')
        df = pd.read_csv(csv_path)
        return dict(zip(df['def_lab'], df['RPTG']))

def group_by_rptg(rptg_dict, bins, labels=None):
    """
    按RPTG区间分组，返回{group_label: [grain_id, ...]}
    bins: list, 如[0, 0.02, 0.04, 0.06]
    labels: list, 如['0-0.02', '0.02-0.04', '0.04-0.06']，可选
    """
    if labels is None:
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    group_dict = {label: [] for label in labels}
    last_label = labels[-1]
    for lab, val in rptg_dict.items():
        try:
            v = float(val)
        except:
            continue
        for i in range(len(bins)-1):
            if (i < len(bins)-2 and bins[i] <= v < bins[i+1]) or (i == len(bins)-2 and bins[i] <= v <= bins[i+1]):
                group_dict[labels[i]].append(lab)
                break
        else:
            # 不在任何区间，归到最后一组
            group_dict[last_label].append(lab)
    return group_dict

def count_group_ratios_from_cycles(cycles, group_dict, group_list, cycle_size):
    """
    统计所有环中各RPTG组颗粒数，返回{group: S_G}, N为环数
    """
    group_counts = {g: 0 for g in group_list}
    N = len(cycles)
    # 反向映射颗粒id到组
    grainid_to_group = {}
    for g in group_list:
        for lab in group_dict.get(g, []):
            grainid_to_group[lab] = g
    for cycle in cycles:
        for gid in cycle:
            group = grainid_to_group.get(gid, None)
            if group in group_counts:
                group_counts[group] += 1
    return group_counts, N

def main():
    parser = argparse.ArgumentParser(description="Calculate average proportion of each RPTG group in cycles of given size and plot trend.")
    parser.add_argument('--cycle_size', default=3, type=int, choices=[3, 4], help='Cycle size (3 or 4)')
    parser.add_argument('--bins', type=float, nargs='+', default=[0, 0.02, 0.04, 0.06], help='RPTG group bins')
    parser.add_argument('--labels', type=str, nargs='+', default=['0-0.02', '0.02-0.04', '0.04-0.06'], help='RPTG group labels')
    args = parser.parse_args() 
    data_dir = '/share/home/202321008879/data/contact_network'
    rptg_dir = '/share/home/202321008879/data/rptg'
    origin_final_results_file = '/share/home/202321008879/data/track_results/load1toorigin/final_results.npz'
    save_dir = '/share/home/202321008879/data/analysis/cycles_rptg_ratio'
    stages = ['origin', 'load1', 'load5', 'load10', 'load15']
    stage_labels = ['0', '1', '5', '10', '15']
    bins = args.bins
    group_list = args.labels
    cycle_size = args.cycle_size
    ratio_df = pd.DataFrame(index=group_list, columns=stage_labels, dtype=float)
    for idx, (stage, label) in enumerate(zip(stages, stage_labels)):
        npz_path = os.path.join(data_dir, f'{stage}', 'cycles.npz')
        if not os.path.exists(npz_path):
            continue
        npz = np.load(npz_path, allow_pickle=True)
        key = f"{cycle_size}_cycles"
        if key not in npz:
            continue
        cycles = npz[key]
        if stage == 'origin':
            rptg_dict = get_stage_rptg_dict(stage, rptg_dir, origin_final_results_file)
        else:
            prev_stage = stages[idx-1]
            rptg_dict = get_stage_rptg_dict(stage, rptg_dir, origin_final_results_file, prev_stage=prev_stage)
        group_dict = group_by_rptg(rptg_dict, bins, group_list)
        group_counts, N = count_group_ratios_from_cycles(cycles, group_dict, group_list, cycle_size)
        for g in group_list:
            ratio_df.loc[g, label] = group_counts[g] / (cycle_size*N) if N > 0 else float('nan')
    # Save data
    out_dir = os.path.join(save_dir, f'{cycle_size}cycle_rptg_ratio')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f'{cycle_size}cycle_rptg_ratio_trend.csv')
    ratio_df.to_csv(out_csv)
    print(f"Cycle size {cycle_size} RPTG group ratio trend data saved to: {out_csv}")
    # Plot
    plt.figure(figsize=(8,6))
    x = list(range(len(stage_labels)))
    for g in group_list:
        plt.plot(x, ratio_df.loc[g], marker='o', label=g)
    plt.xlabel('Strain Stage')
    plt.ylabel('Average Proportion')
    plt.title(f'Average Proportion of Each RPTG Group in {cycle_size}-Cycles')
    plt.xticks(x, stage_labels)
    plt.legend(title='RPTG Group')
    plt.tight_layout()
    out_png = os.path.join(out_dir, f'{cycle_size}cycle_rptg_ratio_trend.png')
    plt.savefig(out_png)
    plt.close()
    print(f"Cycle size {cycle_size} RPTG group ratio trend plot saved to: {out_png}")

if __name__ == '__main__':
    main()
