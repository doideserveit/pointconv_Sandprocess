# 追踪任意阶段任意RPTG组颗粒在上个阶段的来源分布
# 这个来源分布是指：本阶段某个RPTG组的颗粒在上个阶段的RPTG分组情况，
# 如果上个阶段的很多高rptg组不留在本组（只看宏观高rptg组的数量变化，可以发现5-10阶段翻倍，但是来自高rptg组的并没有占50%），会导致高rptg组的占比很低。
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rptg_dir = '/share/home/202321008879/data/rptg'
final_results_dir = '/share/home/202321008879/data/track_results'
save_dir = '/share/home/202321008879/data/analysis/high_rptg_origin'
os.makedirs(save_dir, exist_ok=True)

rptg_bins = [0, 0.02, 0.04, 0.06]
rptg_labels = ['0-0.02', '0.02-0.04', '0.04-0.06']

def group_rptg(val):
    try:
        v = float(val)
        for i in range(len(rptg_bins) - 1):
            if rptg_bins[i] <= v < rptg_bins[i + 1]:
                return rptg_labels[i]
        # 大于等于最后一个bin上限的都归到最后一组
        if v >= rptg_bins[-1]:
            return rptg_labels[-1]
    except:
        return None

def get_rptg_group_labs(stage, prev_stage, group_label):
    """
    获取某阶段指定RPTG组的def_lab列表
    """
    rptg_csv = os.path.join(rptg_dir, f'{stage}to{prev_stage}_RPTG.csv')
    df = pd.read_csv(rptg_csv)
    df['group'] = df['RPTG'].apply(group_rptg)
    labs = df.loc[df['group'] == group_label, 'def_lab'].tolist()
    return labs

def get_deflab_to_reflab(stage, prev_stage):
    """
    获取def_lab到上阶段ref_lab的映射
    """
    final_results_path = os.path.join(final_results_dir, f'{stage}to{prev_stage}', 'final_results.npz')
    data = np.load(final_results_path, allow_pickle=True)
    results = data['results'].item()
    return {def_lab: int(info['predicted_reference_label']) for def_lab, info in results.items()}

def get_reflab_to_group(prev_stage, prev_prev_stage):
    """
    获取上阶段def_lab到RPTG分组的映射
    """
    rptg_csv = os.path.join(rptg_dir, f'{prev_stage}to{prev_prev_stage}_RPTG.csv')
    df = pd.read_csv(rptg_csv)
    df['group'] = df['RPTG'].apply(group_rptg)
    return dict(zip(df['def_lab'], df['group']))

def read_cycles_npz(cycles_npz_file, size=3):
    """读取cycles.npz，返回所有参与size环的def_lab集合"""
    if not os.path.exists(cycles_npz_file):
        return set()
    data = np.load(cycles_npz_file, allow_pickle=True)
    key = f"{size}_cycles"
    if key not in data:
        return set()
    cycles = data[key]
    labs = set()
    for cycle in cycles:
        labs.update(cycle)
    return set(map(int, labs))

def track_rptg_origin(stage, prev_stage, prev_prev_stage, group_label, save_prefix=None):
    """
    追踪指定阶段指定RPTG组颗粒在上阶段的分组分布，并统计三环参与比例
    """
    # 1. 获取本阶段该组颗粒def_lab
    labs = get_rptg_group_labs(stage, prev_stage, group_label)
    # 2. def_lab -> 上阶段ref_lab
    deflab_to_reflab = get_deflab_to_reflab(stage, prev_stage)
    reflabs = [deflab_to_reflab[lab] for lab in labs if lab in deflab_to_reflab]
    # 3. 上阶段ref_lab -> 上阶段RPTG分组
    reflab_to_group = get_reflab_to_group(prev_stage, prev_prev_stage)
    group_counts = {label: 0 for label in rptg_labels}
    group_labs = {label: [] for label in rptg_labels}  # 新增：记录每个来源组的def_lab
    for i, reflab in enumerate(reflabs):
        group = reflab_to_group.get(reflab)
        if group in group_counts:
            group_counts[group] += 1
            group_labs[group].append(labs[i])
    total = sum(group_counts.values())
    group_ratios = {k: (v / total if total > 0 else 0) for k, v in group_counts.items()}

    # 4. 统计三环参与比例
    cycles_npz_file = os.path.join('/share/home/202321008879/data/contact_network', stage, 'cycles.npz')
    tri_cycle_labs = read_cycles_npz(cycles_npz_file, size=3)
    tri_cycle_ratios = {}
    for k in rptg_labels:
        src_labs = set(group_labs[k])
        if not src_labs:
            tri_cycle_ratios[k] = 0
        else:
            tri_cycle_ratios[k] = len(src_labs & tri_cycle_labs) / len(src_labs)

    # 5. 保存和绘图
    if save_prefix is None:
        save_prefix = f'{stage}_{group_label}_origin'
    out_csv = os.path.join(save_dir, f'{save_prefix}_stats.csv')
    df_out = pd.DataFrame({'count': group_counts, 'ratio': group_ratios, 'tri_cycle_ratio': tri_cycle_ratios})
    df_out.to_csv(out_csv)
    plt.figure(figsize=(6,4))
    plt.bar(df_out.index, df_out['ratio'], color='skyblue')
    plt.ylabel('Proportion')
    plt.xlabel(f'RPTG Group in {prev_stage}')
    plt.title(f'Origin of {group_label} Grains in {stage}')
    for i, v in enumerate(df_out['ratio']):
        plt.text(i, v, f"{v:.2%}", ha='center', va='bottom')
    plt.tight_layout()
    out_png = os.path.join(save_dir, f'{save_prefix}_hist.png')
    plt.savefig(out_png)
    plt.close()
    print(f"统计数据已保存到: {out_csv}")
    print(f"三环参与比例: {tri_cycle_ratios}")
    print(f"直方图已保存到: {out_png}")

def collect_cycle_size_freq_by_source(group_labs, size_to_labels):
    """
    对每个本阶段rptg组的每个来源，统计其颗粒在各size环中的出现频率（允许重复计数）。
    返回: {本组: {来源组: {size: freq, ...}, ...}, ...}
    """
    result = {}
    for g, sources in group_labs.items():
        result[g] = {}
        for gg, labs in sources.items():
            labs_set = set(labs)
            size_freq = {}
            total_count = 0
            for size, labels in size_to_labels.items():
                count = sum(1 for lab in labels if lab in labs_set)
                size_freq[size] = count
                total_count += count
            freq_dict = {size: (size_freq[size]/total_count if total_count > 0 else 0) for size in size_freq}
            result[g][gg] = freq_dict
    return result

def track_all_rptg_origins(stage, prev_stage, prev_prev_stage, save_prefix=None):
    """
    统计并绘制本阶段所有RPTG组在上阶段的来源分布（分组柱状图），并统计三环参与比例
    """
    # 1. 获取本阶段所有颗粒及其分组
    rptg_csv = os.path.join(rptg_dir, f'{stage}to{prev_stage}_RPTG.csv')
    df = pd.read_csv(rptg_csv)
    df['group'] = df['RPTG'].apply(group_rptg)
    # 2. def_lab -> 上阶段ref_lab
    deflab_to_reflab = get_deflab_to_reflab(stage, prev_stage)
    # 3. 上阶段ref_lab -> 上阶段RPTG分组
    reflab_to_group = get_reflab_to_group(prev_stage, prev_prev_stage)
    # 4. 统计每个本阶段RPTG组在上阶段的来源分布
    group_counts = {g: {gg: 0 for gg in rptg_labels} for g in rptg_labels}
    group_labs = {g: {gg: [] for gg in rptg_labels} for g in rptg_labels}  # 新增
    for _, row in df.iterrows():
        g = row['group']
        def_lab = row['def_lab']
        if g not in rptg_labels:
            continue
        reflab = deflab_to_reflab.get(def_lab)
        if reflab is None:
            continue
        prev_g = reflab_to_group.get(reflab)
        if prev_g in rptg_labels:
            group_counts[g][prev_g] += 1
            group_labs[g][prev_g].append(def_lab)
    # 5. 计算比例
    group_ratios = {g: {} for g in rptg_labels}
    for g in rptg_labels:
        total = sum(group_counts[g].values())
        for gg in rptg_labels:
            group_ratios[g][gg] = group_counts[g][gg] / total if total > 0 else 0

    # 6. 统计三环参与比例
    cycles_npz_file = os.path.join('/share/home/202321008879/data/contact_network', stage, 'cycles.npz')
    sizes = [3, 4, 5, 6]
    size_to_labels = {}
    if os.path.exists(cycles_npz_file):
        data = np.load(cycles_npz_file, allow_pickle=True)
        for size in sizes:
            key = f"{size}_cycles"
            if key in data:
                cycles = data[key]
                labels = []
                for cycle in cycles:
                    labels.extend(list(cycle))
                size_to_labels[size] = labels

    # 新增：统计每个来源在各size环中的出现频率
    cycle_freq_by_source = collect_cycle_size_freq_by_source(group_labs, size_to_labels)
    # 保存为csv
    freq_rows = []
    for g in rptg_labels:
        for gg in rptg_labels:
            freq_dict = cycle_freq_by_source.get(g, {}).get(gg, {})
            for size in sizes:
                freq = freq_dict.get(size, 0)
                freq_rows.append({'group': g, 'source': gg, 'cycle_size': size, 'freq': freq})
    freq_df = pd.DataFrame(freq_rows)
    if save_prefix is None:
        save_prefix = f'{stage}_all_groups_origin'
    out_freq_csv = os.path.join(save_dir, f'{save_prefix}_cycle_size_freq_by_source.csv')
    freq_df.to_csv(out_freq_csv, index=False)
    print(f"各来源颗粒在不同size环中的出现频率已保存到: {out_freq_csv}")

    # 7. 保存数据
    df_ratio = pd.DataFrame(group_ratios).T  # 行:本阶段组，列:上阶段组
    df_count = pd.DataFrame(group_counts).T

    # 合并百分比和数量和三环比例，列名加后缀
    df_out = pd.concat(
        [df_ratio.add_suffix('_ratio'), df_count.add_suffix('_count')],
        axis=1
    )
    if save_prefix is None:
        save_prefix = f'{stage}_all_groups_origin'
    out_csv = os.path.join(save_dir, f'{save_prefix}_stats.csv')
    df_out.to_csv(out_csv)
    print(f"统计数据已保存到: {out_csv}")
    # 8. 绘图（分组柱状图）
    plt.figure(figsize=(8,5))
    x = np.arange(len(rptg_labels))
    width = 0.25
    for i, prev_g in enumerate(rptg_labels):
        plt.bar(x + i*width, [df_out.loc[g, f'{prev_g}_ratio'] for g in rptg_labels], width=width, label=f'From {prev_g}')
    plt.xticks(x + width, rptg_labels)
    plt.xlabel(f'RPTG Group in {stage}')
    plt.ylabel('Proportion from previous stage')
    plt.title(f'Origin Distribution of RPTG Groups in {stage}')
    plt.legend(title=f'Prev Stage Group ({prev_stage})')
    for i, g in enumerate(rptg_labels):
        for j, prev_g in enumerate(rptg_labels):
            v = df_out.loc[g, f'{prev_g}_ratio']
            plt.text(i + j*width, v, f"{v:.2%}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(save_dir, f'{save_prefix}_hist.png')
    plt.savefig(out_png)
    plt.close()
    print(f"直方图已保存到: {out_png}")

if __name__ == '__main__':
    stages = ['origin', 'load1', 'load5', 'load10', 'load15']
    cur_stage = 'load10'
    previous_stage = stages[stages.index(cur_stage) - 1]
    p_p_stage = stages[stages.index(previous_stage) - 1] if previous_stage != 'origin' else None
    track_all_rptg_origins(stage=cur_stage, prev_stage=previous_stage, prev_prev_stage=p_p_stage)

