# 这个是旧的，被2025.03.04丢弃，这个是对应于单个class存储的H5数据，还得映射的
from joblib import Parallel, delayed
import os
import re
import h5py
import numpy as np
import warnings
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def natural_sort_key(string):
    """
    Key function for natural sorting (e.g., 'class_10' > 'class_2').
    """
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', string)]

def pc_normalize(pc):
    """
    Normalize the point cloud to be centered at origin with a unit bounding sphere.
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Perform farthest point sampling on the input point cloud.
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, n_jobs=4):
        """
        Initialize the dataset.
        Args:
            root: Root directory containing 'train' and 'test' subdirectories.
            npoint: Number of points to sample from each subset.
            split: 'train' or 'test'.
            uniform: Whether to use farthest point sampling.
            normal_channel: Whether to include normals in the input.
            n_jobs: Number of threads to use for parallel loading.
        """
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.split = split
        self.normal_channel = normal_channel

        # 获取数据路径并检查合法性
        self.samples = []  # 所有样本存储在内存中
        split_dir = os.path.join(self.root, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory '{split_dir}' does not exist.")

        # 类别解析
        class_names = sorted(os.listdir(split_dir), key=natural_sort_key)  # 自然排序
        self.classes = {name: idx for idx, name in enumerate(class_names)}  # 类别名称映射到整数索引

        # 加载所有 .h5 文件并并行展开子集
        self.samples = self._load_all_h5_files(split_dir, class_names, n_jobs)

        # 打印加载信息
        print(f"Loaded {len(self.samples)} samples from '{split}' split across {len(self.classes)} classes.")
        print(f"Classes: {self.classes}")

    def _load_all_h5_files(self, split_dir, class_names, n_jobs):
        """
        Load all .h5 files and their subsets into memory using parallel processing.
        """
        datapath = []
        for class_dir in class_names:
            class_path = os.path.join(split_dir, class_dir)
            if os.path.isdir(class_path):
                for h5_file in sorted(os.listdir(class_path)):
                    if h5_file.endswith('.h5'):
                        datapath.append((class_dir, os.path.join(class_path, h5_file)))

        # 并行加载 .h5 文件
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self._load_h5_file)(class_dir, h5_path) for class_dir, h5_path in datapath
        )

        # 将所有子集展开成独立样本
        samples = [item for sublist in results for item in sublist]
        return samples

    def _load_h5_file(self, class_name, h5_path):
        """
        Load a single .h5 file and return its subsets as individual samples.
        """
        try:
            with h5py.File(h5_path, 'r') as f:
                # 根据 split 动态加载数据集
                dataset_name = f"{self.split}_subsets"
                if dataset_name not in f:
                    print(f"Skipping file '{h5_path}': No '{dataset_name}' dataset found.")
                    return []
                subsets = f[dataset_name][:]
                return [(subset, self.classes[class_name]) for subset in subsets]  # 返回数据和其标签
        except Exception as e:
            print(f"Error loading .h5 file at '{h5_path}': {e}")
            return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        point_set, cls = self.samples[index]

        # 是否使用最远点采样
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[:self.npoints, :]

        # 归一化点云
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # 是否包含法向量
        if not self.normal_channel:
            point_set = point_set[:, 0:3]

        return point_set, np.array([cls]).astype(np.int32)


if __name__ == '__main__':
    import torch
    import time

    # 测试数据加载器性能
    start_time = time.time()
    # split=训练'train'或测试'test'
    data = ModelNetDataLoader('I:/sandprocess/data/originbad/fps_subsets', split='train', uniform=False,
                              normal_channel=True, n_jobs=14)
    print(f"Train samples: {len(data)}")
    end_time = time.time()
    print(f"Data loading completed in {end_time - start_time:.2f} seconds.")

    # 测试 DataLoader
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)

    for point, label in DataLoader:
        print(f"Batch points shape: {point.shape}, Batch labels shape: {label.shape}")


# 用torch自带的FPS
# from joblib import Parallel, delayed
# import os
# import re
# import h5py
# import numpy as np
# import warnings
# from torch.utils.data import Dataset
# import torch
# from torch_cluster import fps  # 使用 torch_cluster 提供的最远点采样函数
#
# warnings.filterwarnings('ignore')
#
# def natural_sort_key(string):
#     """
#     Key function for natural sorting (e.g., 'class_10' > 'class_2').
#     """
#     return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', string)]
#
# def pc_normalize(pc):
#     """
#     Normalize the point cloud to be centered at origin with a unit bounding sphere.
#     """
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
#     pc = pc / m
#     return pc
#
# def farthest_point_sample(point, npoint):
#     """
#     使用 PyTorch Cluster 库中的 fps 函数进行最远点采样。
#     Args:
#         point: (N, 6) 输入点云 (xyz + normals)
#         npoint: 要采样的点数。
#     Returns:
#         采样后的点云。
#     """
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     points_xyz = torch.tensor(point[:, :3], dtype=torch.float32).to(device)
#     batch = torch.zeros(points_xyz.shape[0], dtype=torch.long, device=device)  # 单个批次
#
#     sampled_indices = fps(points_xyz, batch, ratio=npoint / points_xyz.shape[0])
#     sampled_points = point[sampled_indices.cpu().numpy(), :]
#
#     return sampled_points
#
# class ModelNetDataLoader(Dataset):
#     def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, n_jobs=4):
#         """
#         Initialize the dataset.
#         Args:
#             root: Root directory containing 'train' and 'test' subdirectories.
#             npoint: Number of points to sample from each subset.
#             split: 'train' or 'test'.
#             uniform: Whether to use farthest point sampling.
#             normal_channel: Whether to include normals in the input.
#             n_jobs: Number of threads to use for parallel loading.
#         """
#         self.root = root
#         self.npoints = npoint
#         self.uniform = uniform
#         self.split = split
#         self.normal_channel = normal_channel
#
#         # 获取数据路径并检查合法性
#         self.samples = []  # 所有样本存储在内存中
#         split_dir = os.path.join(self.root, split)
#         if not os.path.exists(split_dir):
#             raise FileNotFoundError(f"Split directory '{split_dir}' does not exist.")
#
#         # 类别解析
#         class_names = sorted(os.listdir(split_dir), key=natural_sort_key)
#         self.classes = {name: idx for idx, name in enumerate(class_names)}
#
#         # 加载所有 .h5 文件并并行展开子集
#         self.samples = self._load_all_h5_files(split_dir, class_names, n_jobs)
#
#         # 打印加载信息
#         print(f"Loaded {len(self.samples)} samples from '{split}' split across {len(self.classes)} classes.")
#         print(f"Classes: {self.classes}")
#
#     def _load_all_h5_files(self, split_dir, class_names, n_jobs):
#         """
#         Load all .h5 files and their subsets into memory using parallel processing.
#         """
#         datapath = []
#         for class_dir in class_names:
#             class_path = os.path.join(split_dir, class_dir)
#             if os.path.isdir(class_path):
#                 for h5_file in sorted(os.listdir(class_path)):
#                     if h5_file.endswith('.h5'):
#                         datapath.append((class_dir, os.path.join(class_path, h5_file)))
#
#         # 并行加载 .h5 文件
#         results = Parallel(n_jobs=n_jobs, backend="loky")(
#             delayed(self._load_h5_file)(class_dir, h5_path) for class_dir, h5_path in datapath
#         )
#
#         # 将所有子集展开成独立样本
#         samples = [item for sublist in results for item in sublist]
#         return samples
#
#     def _load_h5_file(self, class_name, h5_path):
#         """
#         Load a single .h5 file and return its subsets as individual samples.
#         """
#         try:
#             with h5py.File(h5_path, 'r') as f:
#                 train_subsets = f['train_subsets'][:]
#                 test_subsets = f['test_subsets'][:]
#
#                 # 根据 split 决定加载哪部分数据
#                 subsets = train_subsets if self.split == 'train' else test_subsets
#                 return [(subset, self.classes[class_name]) for subset in subsets]
#         except Exception as e:
#             raise RuntimeError(f"Error loading .h5 file at '{h5_path}': {e}")
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, index):
#         point_set, cls = self.samples[index]
#
#         # 是否使用最远点采样
#         if self.uniform:
#             point_set = farthest_point_sample(point_set, self.npoints)
#         else:
#             point_set = point_set[:self.npoints, :]
#
#         # 归一化点云
#         point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
#
#         # 是否包含法向量
#         if not self.normal_channel:
#             point_set = point_set[:, 0:3]
#
#         return point_set, np.array([cls]).astype(np.int32)
#
#
# if __name__ == '__main__':
#     import time
#
#     # 测试数据加载器性能
#     start_time = time.time()
#     data = ModelNetDataLoader('I:/sandprocess/data/originbad/fps_subsets', split='train', uniform=True,
#                               normal_channel=True, n_jobs=14)
#     print(f"Train samples: {len(data)}")
#     end_time = time.time()
#     print(f"Data loading completed in {end_time - start_time:.2f} seconds.")
#
#     # 测试 DataLoader
#     DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
#
#     for point, label in DataLoader:
#         print(f"Batch points shape: {point.shape}, Batch labels shape: {label.shape}")
