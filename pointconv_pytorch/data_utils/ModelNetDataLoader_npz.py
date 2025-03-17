from joblib import Parallel, delayed
import os
import re
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
        class_names = sorted(os.listdir(split_dir), key=natural_sort_key)
        self.classes = {name: idx for idx, name in enumerate(class_names)}

        # 加载所有 .npz 文件并并行展开子集
        self.samples = self._load_all_npz_files(split_dir, class_names, n_jobs)

        # 打印加载信息
        print(f"Loaded {len(self.samples)} samples from '{split}' split across {len(self.classes)} classes.")
        print(f"Classes: {self.classes}")

    def _load_all_npz_files(self, split_dir, class_names, n_jobs):
        """
        Load all .npz files and their subsets into memory using parallel processing.
        """
        datapath = []
        for class_dir in class_names:
            class_path = os.path.join(split_dir, class_dir)
            if os.path.isdir(class_path):
                for npz_file in sorted(os.listdir(class_path)):
                    if npz_file.endswith('.npz'):
                        datapath.append((class_dir, os.path.join(class_path, npz_file)))

        # 并行加载 .npz 文件
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self._load_npz_file)(class_dir, npz_path) for class_dir, npz_path in datapath
        )

        # 将所有子集展开成独立样本
        samples = [item for sublist in results for item in sublist]
        return samples

    def _load_npz_file(self, class_name, npz_path):
        """
        Load a single .npz file and return its subsets as individual samples.
        """
        try:
            data = np.load(npz_path)
            subsets = data['subsets']
            return [(subset, self.classes[class_name]) for subset in subsets]
        except Exception as e:
            raise RuntimeError(f"Error loading .npz file at '{npz_path}': {e}")

        # 打印加载信息
        print(f"Loaded {len(self.samples)} samples from '{split}' split across {len(self.classes)} classes.")
        print(f"Classes: {self.classes}")

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
    data = ModelNetDataLoader('I:/sandprocess/data/originbad/fps_subsets_npz', split='train', uniform=False,
                              normal_channel=True, n_jobs=14)
    print(f"Train samples: {len(data)}")
    end_time = time.time()
    print(f"Data loading completed in {end_time - start_time:.2f} seconds.")

    # 测试 DataLoader
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)

    for point, label in DataLoader:
        print(f"Batch points shape: {point.shape}, Batch labels shape: {label.shape}")
