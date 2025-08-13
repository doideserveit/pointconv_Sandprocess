# 计算并绘制分组参数均值随阶段变化的趋势图，或者整体均值随阶段变化的趋势图
import os
import pandas as pd
import matplotlib.pyplot as plt

def read_group_param_csv(csv_file):
    """读取单个阶段的分组参数csv，返回group到参数值列表的映射"""
    df = pd.read_csv(csv_file)
    group_param = {}
    for group, group_df in df.groupby('group'):
        group_param[group] = group_df['param_value'].astype(float).tolist()
    return group_param

def trim_extremes(vals, percent=0):
    """去除两端极值，保留中间部分，设置为0时不去除极值"""
    if not vals:
        return vals
    s = pd.Series(vals)
    lower = s.quantile(percent)
    upper = s.quantile(1 - percent)
    trimmed = s[(s >= lower) & (s <= upper)]
    return trimmed.tolist()

def collect_stage_group_means(stage_files):
    """
    输入: {阶段名: csv文件路径}
    输出: DataFrame，index为group，columns为stage，值为均值
    """
    all_groups = set()
    stage_group_means = {}
    for stage, csv_file in stage_files.items():
        group_param = read_group_param_csv(csv_file)
        # 对每组去除极值后再算均值
        means = {g: (sum(trim_extremes(vals))/len(trim_extremes(vals)) if len(trim_extremes(vals))>0 else float('nan')) for g, vals in group_param.items()}
        stage_group_means[stage] = means
        all_groups.update(group_param.keys())
    all_groups = sorted(all_groups)
    df = pd.DataFrame({stage: [stage_group_means[stage].get(g, float('nan')) for g in all_groups] for stage in stage_files})
    df.index = all_groups
    return df

def collect_stage_overall_means(stage_files, trim_percent=0):
    """
    输入: {阶段名: csv文件路径}
    输出: Series，index为stage，值为该阶段所有颗粒的均值（可选去极值）
    """
    means = {}
    for stage, csv_file in stage_files.items():
        df = pd.read_csv(csv_file)
        vals = df['param_value'].astype(float).tolist()
        trimmed = trim_extremes(vals, trim_percent)
        means[stage] = sum(trimmed)/len(trimmed) if trimmed else float('nan')
    return pd.Series(means)

def plot_group_trend(df, overall_means, output_png):
    """绘制每组的均值随阶段变化的折线图，并加上整体均值，横轴为数字"""
    stage_map = {'origin': 0, 'load1': 1, 'load5': 5, 'load10': 10, 'load15': 15}
    x = [stage_map[stage] for stage in df.columns]
    plt.figure(figsize=(8,6))
    for group in df.index:
        plt.plot(x, df.loc[group], marker='o', label=group)
    plt.plot(x, overall_means.values, marker='o', linestyle='--', color='black', label='ALL')
    plt.xlabel('Stage')
    plt.ylabel('Mean Value')
    plt.title('Group Mean Parameter Value Trend')
    plt.xticks(x)  # 只显示数字
    plt.legend(title='Group')
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

def plot_overall_trend(series, output_png):
    """绘制所有颗粒均值随阶段变化的折线图，横轴为数字"""
    stage_map = {'origin': 0, 'load1': 1, 'load5': 5, 'load10': 10, 'load15': 15}
    x = [stage_map[stage] for stage in series.index]
    plt.figure(figsize=(8,6))
    plt.plot(x, series.values, marker='o')
    plt.xlabel('Stage')
    plt.ylabel('Mean Value')
    plt.title('Overall Mean Parameter Value Trend')
    plt.xticks(x)  # 只显示数字
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

def main():
    # 参数配置
    data_dir = '/share/home/202321008879/data/analysis/clustering'
    stages = ['origin', 'load1', 'load5', 'load10', 'load15']
    param_name = 'clustering'
    # 构造每个阶段的csv文件路径
    stage_files = {stage: os.path.join(data_dir, f'{stage}_{param_name}_by_rptg.csv') for stage in stages}
    # 1. 读取并计算均值
    df = collect_stage_group_means(stage_files)
    # 2. 计算整体均值
    overall_means = collect_stage_overall_means(stage_files, trim_percent=0)
    # 3. 保存分组均值csv
    output_csv = os.path.join(data_dir, f'group_{param_name}_mean_trend.csv')
    df.to_csv(output_csv)
    print(f"分组均值趋势已保存到: {output_csv}")
    # 4. 保存整体均值csv
    overall_csv = os.path.join(data_dir, f'overall_{param_name}_mean_trend.csv')
    overall_means.to_csv(overall_csv, header=['mean'])
    print(f"整体均值趋势已保存到: {overall_csv}")
    # 5. 绘图（分组+整体均值）
    output_png = os.path.join(data_dir, f'group_{param_name}_mean_trend.png')
    plot_group_trend(df, overall_means, output_png)
    print(f"分组均值趋势折线图已保存到: {output_png}")

    # 6. 绘图（整体均值）
    overall_png = os.path.join(data_dir, f'overall_{param_name}_mean_trend.png')
    plot_overall_trend(overall_means, overall_png)
    print(f"整体均值趋势折线图已保存到: {overall_png}")

if __name__ == '__main__':
    main()
