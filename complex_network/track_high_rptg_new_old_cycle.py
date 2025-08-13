import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_high_rptg_labs(rptg_csv, low=0.4, high=0.6):
    df = pd.read_csv(rptg_csv)
    return set(df.loc[(df['RPTG'] >= low) & (df['RPTG'] <= high), 'def_lab'])

def get_deflab_to_reflab(final_results_file):
    data = np.load(final_results_file, allow_pickle=True)
    results = data['results'].item()
    return {def_lab: int(info['predicted_reference_label']) for def_lab, info in results.items()}

def get_cycle_labels(cycles_npz_file, sizes=[3,4,5,6]):
    data = np.load(cycles_npz_file, allow_pickle=True)
    size_to_labels = {}
    for size in sizes:
        key = f"{size}_cycles"
        if key in data:
            cycles = data[key]
            labels = []
            for cycle in cycles:
                labels.extend(list(cycle))
            size_to_labels[size] = labels
    return size_to_labels

def count_cycle_freq(lab_set, size_to_labels):
    freq = {}
    total = 0
    for size, labels in size_to_labels.items():
        count = sum(1 for lab in labels if lab in lab_set)
        freq[size] = count
        total += count
    freq_ratio = {size: (freq[size]/total if total>0 else 0) for size in freq}
    return freq, freq_ratio

if __name__ == '__main__':
    data_dir = '/share/home/202321008879/data'
    rptg_dir = os.path.join(data_dir, 'rptg')
    cycles_npz_file = os.path.join(data_dir, 'contact_network', 'load10', 'cycles.npz')
    save_dir = os.path.join(data_dir, 'analysis', 'high_rptg_new_old_cycle')
    os.makedirs(save_dir, exist_ok=True)

    # 1. 获取load5高RPTG颗粒def_lab
    labs_5 = get_high_rptg_labs(os.path.join(rptg_dir, 'load5toload1_RPTG.csv'))
    # 2. 获取load10高RPTG颗粒def_lab
    labs_10 = get_high_rptg_labs(os.path.join(rptg_dir, 'load10toload5_RPTG.csv'))

    # 3. 追踪load5高RPTG颗粒到load10的def_lab
    final_results_10_5 = os.path.join(data_dir, 'track_results', 'load10toload5', 'final_results.npz')
    deflab10_to_reflab5 = get_deflab_to_reflab(final_results_10_5)
    # 反向映射：load5 ref_lab -> load10 def_lab（可能一对多）
    reflab5_to_deflab10 = {}
    for def10, ref5 in deflab10_to_reflab5.items():
        reflab5_to_deflab10.setdefault(ref5, []).append(def10)

    # 4. 追踪load5高RPTG颗粒在load10的def_lab集合
    labs_5_in_10 = set()
    for lab5 in labs_5:
        labs_5_in_10.update(reflab5_to_deflab10.get(lab5, []))

    # 5. 新加入 = load10高RPTG组 - load5高RPTG组追踪到的
    new_labs = labs_10 - labs_5_in_10
    print(f"新加入的高RPTG颗粒数量: {len(new_labs)}")
    # 6. 一直高 = load10高RPTG组 ∩ load5高RPTG组追踪到的
    old_labs = labs_10 & labs_5_in_10
    print(f"一直高的高RPTG颗粒数量: {len(old_labs)}")
    # 7. 退出高 = load5高RPTG组追踪到的 - load10高RPTG组
    exit_labs = labs_5_in_10 - labs_10
    print(f"退出高的高RPTG颗粒数量: {len(exit_labs)}")

    # 8. 获取3、4、5、6环所有颗粒标签
    size_to_labels = get_cycle_labels(cycles_npz_file, sizes=[3,4,5,6])

    # 9. 统计频率
    freq_new, ratio_new = count_cycle_freq(new_labs, size_to_labels)
    freq_old, ratio_old = count_cycle_freq(old_labs, size_to_labels)
    freq_exit, ratio_exit = count_cycle_freq(exit_labs, size_to_labels)

    # 10. 保存数据
    df_out = pd.DataFrame({
        'cycle_size': list(size_to_labels.keys())*3,
        'type': ['new']*len(size_to_labels) + ['old']*len(size_to_labels) + ['exit']*len(size_to_labels),
        'count': [freq_new[k] for k in size_to_labels] + [freq_old[k] for k in size_to_labels] + [freq_exit[k] for k in size_to_labels],
        'ratio': [ratio_new[k] for k in size_to_labels] + [ratio_old[k] for k in size_to_labels] + [ratio_exit[k] for k in size_to_labels]
    })
    out_csv = os.path.join(save_dir, 'high_rptg_new_old_exit_cycle_stats.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"统计数据已保存到: {out_csv}")

    # 11. 柱状图绘制
    plt.figure(figsize=(8,6))
    bar_width = 0.22
    x = np.arange(len(size_to_labels))
    types = ['new', 'old', 'exit']
    labels = ['New in load10', 'Kept high RPTG', 'Exit high RPTG']
    colors = ['#4C72B0', '#55A868', '#C44E52']
    for i, (t, label, color) in enumerate(zip(types, labels, colors)):
        ratios = df_out[df_out['type']==t].set_index('cycle_size')['ratio']
        plt.bar(x + i*bar_width, ratios.values, width=bar_width, label=label, color=color)
        for j, v in enumerate(ratios.values):
            plt.text(x[j] + i*bar_width, v, f"{v:.2%}", ha='center', va='bottom', fontsize=9)
    plt.xlabel('Cycle Size')
    plt.ylabel('Frequency')
    plt.title('Cycle Participation Frequency: New/Old/Exit High RPTG Grains in Load10')
    plt.xticks(x + bar_width, list(size_to_labels.keys()))
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(save_dir, 'high_rptg_new_old_exit_cycle_freq.png')
    plt.savefig(out_png)
    plt.close()
    print(f"频率柱状图已保存到: {out_png}")
