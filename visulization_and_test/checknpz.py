import numpy as np
import matplotlib.pyplot as plt

# 读取Npz文件中的键
file = np.load('/share/home/202321008879/data/track_results/load1_aitoorigin_ai/load1_track_results(printdistancerange).npz', allow_pickle=True)
results = file['results']

# 筛选 avg_prediction 大于 60% 的颗粒
probs = np.array([float(a[2]) for a in results])
filtered_results = [a for a, prob in zip(results, probs) if prob >= 0.6]
filtered_probs = np.array([float(a[2]) for a in filtered_results])  # 筛选后的 avg_prediction

# 确认筛选逻辑
print(f"Number of predictions greater than 0.6: {len(filtered_results)}")
print(f">60% percentage: {len(filtered_results) / len(results) * 100:.2f}%")

# 统计 distance_range 的分布
count_62d50 = np.sum(np.array([a[10] for a in filtered_results]) == '<2D50')
count_62to4 = np.sum(np.array([a[10] for a in filtered_results]) == '2D50-4D50')
count_64d50 = np.sum(np.array([a[10] for a in filtered_results]) == '>4D50')
print(f"Number of predictions greater than 0.6 and distance_range <2D50: {count_62d50}, percentage: {count_62d50 / len(filtered_results) * 100:.2f}%")
print(f"Number of predictions greater than 0.6 and distance_range 2D50-4D50: {count_62to4}, percentage: {count_62to4 / len(filtered_results) * 100:.2f}%")
print(f"Number of predictions greater than 0.6 and distance_range >4D50: {count_64d50}, percentage: {count_64d50 / len(filtered_results) * 100:.2f}%")

# # 打印 avg_prediction 大于 60% 且 distance_range 为 >4D50 的具体信息
# for i, (def_lab, ref_lab) in enumerate(zip(np.array([a[0] for a in filtered_results]), np.array([a[1] for a in filtered_results]))):
#     if np.array([a[10] for a in filtered_results])[i] == '>4D50':
#         print(f"def_lab: {def_lab}, ref_lab: {ref_lab}, avg_prediction: {filtered_probs[i]}, "
#               f"displacement: {np.array([b[9] for b in filtered_results])[i]}, "
#               f"distance_range: {np.array([a[10] for a in filtered_results])[i]}, "
#               f"vol_error: {np.array([c[5] for c in filtered_results])[i]}")

# 对筛选后的颗粒生成 vol_error 的直方图
vol_errors = np.array([float(a[5]) for a in filtered_results])  # 确保转换为浮点数

# 筛选并打印 vol_error > 100 的结果
vol_errors_above_100 = vol_errors[vol_errors > 100]
print(f"Number of volume errors > 100: {len(vol_errors_above_100)}")

# 检查数据范围，分别显示 vol_error 在 0-10，10-20，...，90-100 以及大于 100 的百分比
bins = np.arange(0, 110, 10)
hist, _ = np.histogram(vol_errors, bins=bins)
percentages = hist / len(vol_errors) * 100
print("Volume error percentages in bins:")
for i in range(len(bins) - 1):
    print(f"Range {bins[i]}-{bins[i + 1]}: {percentages[i]:.2f}%")

# 筛选 vol_error > 30 的颗粒
vol_error_above_30 = vol_errors > 30

# 计算 distance_range 为 2D50-4D50 和 >4D50 的占比
distance_range = np.array([a[10] for a in filtered_results])
count_2d50_4d50 = np.sum(vol_error_above_30 & (distance_range == '2D50-4D50'))
count_above_4d50 = np.sum(vol_error_above_30 & (distance_range == '>4D50'))
total_above_30 = np.sum(vol_error_above_30)

percentage_2d50_4d50 = (count_2d50_4d50 / total_above_30) * 100
percentage_above_4d50 = (count_above_4d50 / total_above_30) * 100

print(f"Percentage of particles with vol_error > 30 and distance_range 2D50-4D50: {percentage_2d50_4d50:.2f}%")
print(f"Percentage of particles with vol_error > 30 and distance_range >4D50: {percentage_above_4d50:.2f}%")

# 打印 vol_error > 20 且 distance_range 为 <2D50 的颗粒标签的个数
count_vol_error_30_2d50 = np.sum((vol_errors > 20) & (np.array([a[10] for a in filtered_results]) == '<2D50'))
print(f"Number of particles with vol_error > 20 and distance_range <2D50: {count_vol_error_30_2d50}")
# 打印 vol_error < 30 且 distance_range 为 <2D50 的颗粒标签占总标签个数的百分比
count_vol_error_30_2d50 = np.sum((vol_errors < 30) & (np.array([a[10] for a in filtered_results]) == '<2D50'))
total_count = len(np.array([a[0] for a in results]))
print(f"Percentage of particles with vol_error < 30 and distance_range <2D50: {count_vol_error_30_2d50} / {total_count} : {count_vol_error_30_2d50 / total_count * 100:.2f}%")


# 打印vol_error > 20的颗粒标签，且distance_range为<2D50
print("Volume error > 20 for predictions > 0.6 and distance range < 2D50:")

for i, (def_lab, ref_lab) in enumerate(zip(np.array([a[0] for a in filtered_results]), np.array([a[1] for a in filtered_results]))):
    if np.array([a[10] for a in filtered_results])[i] == '<2D50' and vol_errors[i] > 20:
        print(f"def_lab: {def_lab}, ref_lab: {ref_lab}, avg_prediction: {filtered_probs[i]}, voul_error: {vol_errors[i]}")

# # 绘制直方图
# plt.figure(figsize=(10, 6))
# plt.hist(vol_errors, bins=20, color='blue', alpha=0.7, density=True)
# plt.xlabel('Volume Error')
# plt.ylabel('Frequency')
# plt.title('Histogram of Volume Error for Predictions > 0.6')
# plt.grid()
# plt.tight_layout()
# plt.savefig('volume_error_histogram.png')

# # 绘制直方图（x 坐标为对数，y 坐标为百分比）
# plt.figure(figsize=(10, 6))
# hist, bins, _ = plt.hist(vol_errors, bins=20, color='blue', alpha=0.7, density=True)
# y_percentages = hist / hist.sum() * 100  # 将 y 坐标转换为百分比
# plt.clf()  # 清除当前图形以重新绘制

# plt.bar(bins[:-1], y_percentages, width=np.diff(bins), align='edge', color='blue', alpha=0.7)
# plt.xscale('log')  # 设置 x 轴为对数坐标
# plt.xlabel('Volume Error (Log Scale)')
# plt.ylabel('Percentage (%)')
# plt.title('Histogram of Volume Error for Predictions > 0.6')
# plt.ylim(0, 100)  # 限制 y 坐标范围为 0-100
# plt.grid()
# plt.tight_layout()
# plt.savefig('volume_error_histogram_log_percentage.png')



