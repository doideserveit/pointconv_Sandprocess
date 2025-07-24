# 该脚本用于比较设定了不同倍数的D50的初始匹配结果，读取npz文件里的锚点颗粒，比较多出来的颗粒
import numpy as np
import os

npz_dir1 = '/share/home/202321008879/data/track_results/load15toload10/load15toload10_initmatch_4d50.npz'
npz_dir2 = '/share/home/202321008879/data/track_results/load15toload10/load15toload10_initmatch_6d50.npz'

data1 = np.load(npz_dir1, allow_pickle=True)
data2 = np.load(npz_dir2, allow_pickle=True)
anchor1 = data1['anchor_particles'].item()
anchor2 = data2['anchor_particles'].item()

set1 = set(anchor1.keys())
set2 = set(anchor2.keys())

only_in_1 = set1 - set2
only_in_2 = set2 - set1
print(f"csv1路径：{npz_dir1}")
print(f"csv2路径：{npz_dir2}")
print(f"Only in {npz_dir1}: {len(only_in_1)} particles")
print("锚点标签：预测标签")
for label in sorted(only_in_1):
    print(f"{label}: {anchor1[label]['predicted_reference_label']}")
print(f"Only in {npz_dir2}: {len(only_in_2)} particles")
print("锚点标签：预测标签")
for label in sorted(only_in_2):
    print(f"{label}: {anchor2[label]['predicted_reference_label']}")

