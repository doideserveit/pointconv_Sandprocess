# 检查H5数据中的特定标签和子集
import  h5py
with h5py.File('/share/home/202321008879/data/h5data/load1_selpar/all_eval_fps_subsets.h5', 'r') as f:
    subsets = f['subsets'][:]  # 形状 (N, num_points, 6), N=batch_size
    orig_labels = f['labels'][:]
print(subsets)