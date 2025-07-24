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

def group_by_rptg(rptg_dict, bins):
    """按RPTG区间分组，返回区间到def_lab列表的映射"""
    groups = {f"{bins[i]}-{bins[i+1]}": [] for i in range(len(bins)-1)}
    for lab, val in rptg_dict.items():
        try:
            val = float(val)
        except:
            continue
        for i in range(len(bins)-1):
            if bins[i] <= val < bins[i+1]:
                groups[f"{bins[i]}-{bins[i+1]}"].append(lab)
                break
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

def collect_param_by_group(groups, def2ref, param_dict):
    """按分组收集复杂网络参数值（以ref_lab为主）"""
    group_param = {}
    for group, labs in groups.items():
        vals = []
        for def_lab in labs:
            ref_lab = def2ref.get(def_lab)
            if ref_lab is not None and ref_lab in param_dict:
                vals.append(param_dict[ref_lab])
        group_param[group] = vals
    return group_param

def save_group_param_csv(group_param, output_file):
    """保存分组参数值到CSV"""
    with open(output_file, 'w') as f:
        f.write('group,param_value\n')
        for group, vals in group_param.items():
            for v in vals:
                f.write(f"{group},{v}\n")

def plot_kde(group_param, param_name, output_png):
    """绘制分组KDE核密度估计图"""
    plt.figure(figsize=(8,6))
    for group, vals in group_param.items():
        if len(vals) > 0:
            sns.kdeplot(vals, label=group)
    plt.xlabel(param_name)
    plt.ylabel('Density')
    plt.title(f'{param_name} KDE by RPTG Group')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

def main():
    # 参数配置（可改为argparse）
    stage = 'load1'  # 当前阶段
    ref_stage = 'origin'  # 参考阶段
    param_name = 'degree' # 复杂网络参数名
    bins = [0,0.1,0.2,0.3,0.4,0.5,0.6] # RPTG分组区间

    data_dir = '/share/home/202321008879/data'
    rptg_file = os.path.join(data_dir, 'rptg', f'{stage}to{ref_stage}_RPTG.csv')
    final_results_file = os.path.join(data_dir, 'track_results', f'{stage}to{ref_stage}', 'final_results.npz')
    param_file = os.path.join(data_dir, 'contact_network', stage, f'{param_name}.csv')
    output_csv = os.path.join(data_dir, 'analysis', f'{stage}_{param_name}_by_rptg.csv')
    output_png = os.path.join(data_dir, 'analysis', f'{stage}_{param_name}_by_rptg.png')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 1. 读取RPTG
    rptg_dict = read_rptg(rptg_file)
    # 2. 分组
    groups = group_by_rptg(rptg_dict, bins)
    # 3. 读取def_lab到ref_lab映射
    def2ref = read_final_results(final_results_file)
    # 4. 读取复杂网络参数
    param_dict = read_network_param(param_file)
    # 5. 按分组收集参数值
    group_param = collect_param_by_group(groups, def2ref, param_dict)
    # 6. 输出CSV
    save_group_param_csv(group_param, output_csv)
    print(f"分组参数值已保存到: {output_csv}")
    # 7. 绘图
    plot_kde(group_param, param_name, output_png)
    print(f"KDE图已保存到: {output_png}")

if __name__ == '__main__':
    main()