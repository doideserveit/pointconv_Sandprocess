import  h5py
with h5py.File('I:/sandprocess/data/originnew/all_test_fps_subsets.h5', 'r') as f:
    subsets = f['subsets'][:]  # 形状 (N, num_points, 6), N=batch_size
    orig_labels = f['labels'][:]
print(subsets)