# 检查H5数据中的有多少个标签
import  h5py
import numpy as np
with h5py.File('/share/home/202321008879/data/h5data/load1_selpar/nomapping/all_eval_fps_subsets.h5', 'r') as f:
    # 获取所有的标签
    labels = f['labels'][:]
    # 获取所有的子集
    subsets = f['subsets'][:]
    # 打印标签和子集的数量
    print("Labels:", labels)
    print("Number of unique labels:", len(set(labels)))
    print("Subsets shape:", subsets.shape)

