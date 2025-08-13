# 这个脚本是分支版本，该版本中origin阶段不分组，同时删除了diff的计算
# 这个脚本分析各阶段的复杂网络参数（如度、聚类系数等）的分布情况
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_rptg(rptg_file):
    """读取RPTG文件，返回def_lab到RPTG值的映射"""
    df = pd.read_csv(rptg_file)
    return dict(zip(df['def_lab'], df['RPTG']))

def get_stage_rptg_dict(stage, rptg_file, final_results_file):
    """
    获取当前阶段颗粒的rptg值映射。
    - 对于origin阶段，需将load1toorigin的rptg值通过final_results反向映射到ref_lab，并对多个def_lab取均值。
    - 其它阶段直接读取rptg文件。
    返回: {lab: rptg}
    """
    if stage == 'origin':
        # 读取load1toorigin的rptg
        rptg_df = pd.read_csv(rptg_file)
        rptg_map = dict(zip(rptg_df['def_lab'], rptg_df['RPTG']))
        # 读取final_results，反向映射ref_lab->def_lab
        data = np.load(final_results_file, allow_pickle=True)
        results = data['results'].item()
        ref2defs = {}
        for def_lab, info in results.items():
            ref_lab = int(info['predicted_reference_label'])
            ref2defs.setdefault(ref_lab, []).append(def_lab)
        # 对每个ref_lab，取其所有def_lab的rptg均值
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
        # 其它阶段直接读取
        rptg_df = pd.read_csv(rptg_file)
        return dict(zip(rptg_df['def_lab'], rptg_df['RPTG']))

def group_by_rptg(rptg_dict, bins):
    """按RPTG区间分组，返回区间到lab列表的映射"""
    groups = {f"{bins[i]}-{bins[i+1]}": [] for i in range(len(bins)-1)}
    for lab, val in rptg_dict.items():
        try:
            val = float(val)
        except:
            continue
        assigned = False
        for i in range(len(bins)-2):
            if bins[i] <= val < bins[i+1]:
                groups[f"{bins[i]}-{bins[i+1]}"].append(lab)
                assigned = True
                break
        # 特别处理最后一组：将大于最后一个区间的点都归入最后一组
        if not assigned and bins[-2] <= val:
            groups[f"{bins[-2]}-{bins[-1]}"].append(lab)
    return groups

def read_final_results(final_results_file):
    """读取final_results.npz，返回def_lab到ref_lab的映射"""
    data = np.load(final_results_file, allow_pickle=True)
    results = data['results'].item()
    return {int(def_lab): int(info['predicted_reference_label']) for def_lab, info in results.items()}

def read_network_param(param_file):
    """读取复杂网络参数文件，返回lab到参数值的映射"""
    df = pd.read_csv(param_file)
    return dict(zip(df['node'], df[df.columns[1]]))

def collect_param_by_group(groups, param_dict):
    """按分组收集复杂网络参数值（直接用当前阶段的lab）"""
    group_param = {}
    for group, labs in groups.items():
        vals = []
        for lab in labs:
            if lab in param_dict:
                vals.append(param_dict[lab])
        group_param[group] = vals
    return group_param

def save_group_param_csv(group_param, output_file):
    """保存分组参数值到CSV"""
    with open(output_file, 'w') as f:
        f.write('group,param_value\n')
        for group, vals in group_param.items():
            for v in vals:
                f.write(f"{group},{v}\n")

def plot_kde(group_param, param_name, output_png, xlim_min=None, xlim_max=None):
    """绘制分组KDE核密度估计图，可设定x轴最小/最大值"""
    plt.figure(figsize=(8,6))
    total_count = sum(len(vals) for vals in group_param.values())
    all_vals = []
    all_groups = []
    group_sizes = {}
    for group, vals in group_param.items():
        all_vals.extend(vals)
        all_groups.extend([group] * len(vals))
        group_sizes[group] = len(vals)
    df = pd.DataFrame({'param': all_vals, 'group': all_groups})
    ax = sns.kdeplot(data=df, x='param', hue='group', common_norm=True)
    # 获取seaborn自动生成的legend对象
    legend = ax.get_legend()
    if legend is not None:
        labels = [t.get_text() for t in legend.get_texts()]
        new_labels = []
        for group in labels:
            n = group_sizes.get(group, 0)
            proportion = n / total_count if total_count > 0 else 0
            new_labels.append(f"{group} (n={n}, {proportion:.1%})")
        # 替换legend文本
        for t, new_label in zip(legend.get_texts(), new_labels):
            t.set_text(new_label)
    plt.xlabel(param_name)
    plt.ylabel('Density (with group normalization)')
    plt.title(f'{param_name} KDE by RPTG Group')
    if xlim_min is not None or xlim_max is not None:
        plt.xlim(left=xlim_min, right=xlim_max)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

def plot_group_line(group_param, param_name, output_png, xlim_min=None, xlim_max=None):
    """
    绘制分组折线图（适合离散参数如度），可设定x轴最小/最大值
    非origin阶段，图例标注每组百分比
    """
    plt.figure(figsize=(8,6))
    total_count = sum(len(vals) for vals in group_param.values())
    group_sizes = {group: len(vals) for group, vals in group_param.items()}
    show_percent = len(group_param) > 1  # origin只有一个组，不显示百分比
    for group, vals in group_param.items():
        if len(vals) == 0:
            continue
        value_counts = pd.Series(vals).value_counts().sort_index()
        global_freq = value_counts / total_count # 计算全局频率
        if show_percent:
            n = group_sizes[group]
            proportion = n / total_count if total_count > 0 else 0
            label = f"{group} (n={n}, {proportion:.1%})"
        else:
            label = group
        plt.plot(global_freq.index, global_freq.values, marker='o', label=label)
    plt.xlabel(param_name)
    plt.ylabel('Count')
    plt.title(f'{param_name} Distribution by RPTG Group')
    plt.legend()
    if xlim_min is not None or xlim_max is not None:
        plt.xlim(left=xlim_min, right=xlim_max)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

def parser():
    """解析命令行参数（可选）"""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze complex network parameters by RPTG groups.')
    parser.add_argument('--stage', type=str, required=True, help='Current stage (e.g., load1, origin)')
    parser.add_argument('--param_name', type=str, required=True, help='Complex network parameter name (e.g., degree)')
    parser.add_argument('--bins', type=float, nargs='+', default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], help='RPTG bins for grouping')
    parser.add_argument('--xlim_min', type=float, default=None, help='Minimum x-axis value for plots')
    parser.add_argument('--xlim_max', type=float, default=None, help='Maximum x-axis value for plots')
    return parser.parse_args()

def main():
    args = parser()
    stage_list = ['origin', 'load1', 'load5', 'load10', 'load15']
    stage = args.stage  # 当前阶段
    try:
        stage_list.index(stage)
    except ValueError:
        raise ValueError(f"Invalid stage: {stage}. Must be one of {stage_list}.")
    ref_stage = stage_list[stage_list.index(stage)-1] if stage != 'origin' else None  # 参考阶段
    param_name = args.param_name # 复杂网络参数名
    bins = args.bins # RPTG分组区间
    xlim_min = args.xlim_min
    xlim_max = args.xlim_max
    data_dir = '/share/home/202321008879/data'
    rptg_file = os.path.join(data_dir, 'rptg', f'load1toorigin_RPTG.csv') if stage == 'origin' else os.path.join(data_dir, 'rptg', f'{stage}to{ref_stage}_RPTG.csv')
    final_results_file = os.path.join(data_dir, 'track_results', f'load1toorigin', 'final_results.npz') if stage == 'origin' else os.path.join(data_dir, 'track_results', f'{stage}to{ref_stage}', 'final_results.npz')
    param_file = os.path.join(data_dir, 'contact_network', stage, f'{param_name}.csv')
    output_csv = os.path.join(data_dir, 'analysis', param_name, f'{stage}_{param_name}_by_rptg.csv')
    output_png = os.path.join(data_dir, 'analysis', param_name, f'{stage}_{param_name}_by_rptg.png')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print(f"分析阶段: {stage}, 参考阶段: {ref_stage}, 参数名: {param_name}, 分组区间: {bins}")
    print(f"使用的RPTG文件: {rptg_file}， final_results文件: {final_results_file}, 参数文件: {param_file}")
    
    # 1. 获取rptg映射
    rptg_dict = get_stage_rptg_dict(stage, rptg_file, final_results_file)
    # 2. 分组
    if stage == 'origin':
        # origin阶段不分组，全部归为一组
        groups = {'all': list(rptg_dict.keys())}
    else:
        groups = group_by_rptg(rptg_dict, bins)
    # 3. 读取def_lab到ref_lab映射
    def2ref = read_final_results(final_results_file)
    # 4. 读取复杂网络参数
    param_dict = read_network_param(param_file)
    # 5. 按分组收集参数值（直接用lab）
    group_param = collect_param_by_group(groups, param_dict)
    # 6. 输出CSV
    save_group_param_csv(group_param, output_csv)
    print(f"分组参数值已保存到: {output_csv}")
    # 7. 绘图
    plot_kde(group_param, param_name, output_png, xlim_min=xlim_min, xlim_max=xlim_max)
    print(f"KDE图已保存到: {output_png}")
    # plot_group_line(group_param, param_name, output_png, xlim_min=xlim_min, xlim_max=xlim_max)
    # print(f"分组折线图已保存到: {output_png}")

if __name__ == '__main__':
    main()