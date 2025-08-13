# 一个size一个图，横轴阶段，纵轴是各组RPTG颗粒参与该size环的频率（分母是该RPTG组以颗粒计数的参与的环数）
# 对每个RPTG分组计算其颗粒在n环中的出现次数占该组颗粒在所有大小环中的总出现次数的比值
# 这样RPTG组内，可以比较颗粒参与哪类环比较多
# 同时不同RPTG组之间，可以比较哪些组颗粒更倾向于参与某类环（因为环的所有颗粒都属于RPTG组，所以同一幅图中不同RPTG组的频率值之和是1）
import os
import pandas as pd
import matplotlib.pyplot as plt

def collect_cycle_group_freqs(data_dir, stages, cycle_sizes):
    """
    读取所有阶段的环size频率csv，返回
    {cycle_size: DataFrame(index=rptg_group, columns=stages, values=frequency)}
    """
    # {cycle_size: {group: {stage: freq}}}
    freq_dict = {size: {} for size in cycle_sizes}
    for stage in stages:
        csv_path = os.path.join(data_dir, f'{stage}_cycles_freq_by_rptg.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for size in cycle_sizes:
            sub = df[df['cycle_size'] == size]
            for _, row in sub.iterrows():
                group = row['group']
                freq = row['frequency']
                freq_dict[size].setdefault(group, {})[stage] = freq
    # 转为DataFrame
    size_to_df = {}
    for size in cycle_sizes:
        # 行：group，列：stage
        groups = sorted(freq_dict[size].keys())
        data = {stage: [freq_dict[size][g].get(stage, float('nan')) for g in groups] for stage in stages}
        df = pd.DataFrame(data, index=groups)
        size_to_df[size] = df
    return size_to_df

def plot_cycle_group_trend(df, cycle_size, output_png):
    """
    绘制某个环size在各rptg组内随阶段变化的趋势图
    """
    x = list(range(len(df.columns)))
    plt.figure(figsize=(8,6))
    for group in df.index:
        plt.plot(x, df.loc[group], marker='o', label=group)
    plt.xlabel('Stage')
    plt.ylabel('Frequency')
    plt.title(f'Cycle Size {cycle_size} Frequency Trend by RPTG Group')
    plt.xticks(x, df.columns)
    plt.legend(title='RPTG Group')
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

def main():
    data_dir = '/share/home/202321008879/data/analysis/cycles'
    save_dir = '/share/home/202321008879/data/analysis'
    stages = ['origin', 'load1', 'load5', 'load10', 'load15']
    cycle_sizes = [3, 4, 5, 6]
    out_dir = os.path.join(save_dir, 'cycle_group_trend')
    os.makedirs(out_dir, exist_ok=True)
    size_to_df = collect_cycle_group_freqs(data_dir, stages, cycle_sizes)
    for size, df in size_to_df.items():
        output_png = os.path.join(out_dir, f'cycle{size}_group_trend.png')
        plot_cycle_group_trend(df, size, output_png)
        print(f"环{size}分组频率趋势图已保存到: {output_png}")
        # 保存数据到csv
        output_csv = os.path.join(out_dir, f'cycle{size}_group_trend.csv')
        df.to_csv(output_csv)
        print(f"环{size}分组频率趋势数据已保存到: {output_csv}")

if __name__ == '__main__':
    main()
