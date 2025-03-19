# 这个版本2025.03.04，是对应于所有颗粒的所有子集都保存到一个训练或测试h5文件的，同时还加入了标签
# 由于标签是从1开始，而训练脚本里计算损失函数loss = F.nll_loss(pred, target.long())要求标签都是从0开始，因此还是要进行映射
# 同时这个脚本还可以检测数据集的子集个数，以及每个子集的点云个数
import os
import numpy as np
import h5py
import warnings
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

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
    def __init__(self, root, npoint=1200, split='train', uniform=False, normal_channel=True, n_jobs=4, label_map=None, num_classes=None):
        """
        Initialize the dataset.
        Args:
            root: Directory containing the combined h5 file (e.g. "all_train_fps_subsets.h5" for split 'train').
            npoint: Number of points to sample from each subset.
            split: 'train' or 'test'.
            uniform: Whether to use farthest point sampling.
            normal_channel: Whether to include normals in the input.
            n_jobs: 保留参数，保持原有接口（本版本不使用并行加载）。
            label_map: Label mapping, passed from the training script if using a pre-trained model.
            num_classes: Number of classes, passed from the training script if using a pre-trained model.
        """
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.split = split
        self.normal_channel = normal_channel
        self.label_map = label_map
        self.num_classes = num_classes

        # 构建合并 h5 文件的路径
        h5_filename = os.path.join(self.root, f"all_{split}_fps_subsets.h5")
        if not os.path.exists(h5_filename):
            raise FileNotFoundError(f"Combined h5 file '{h5_filename}' does not exist.")

        # 从 h5 文件中加载数据和原始标签
        with h5py.File(h5_filename, 'r') as f:
            subsets = f['subsets'][:]   # 形状 (N, num_points, 6), N=batch_size
            orig_labels = f['labels'][:]
            print(f"Loaded subsets with shape {subsets.shape}")
            print(f"Loaded original labels with shape {orig_labels.shape}")  # 形状 (N,)，原始标签

        # 如果提供了 label_map 和 num_classes（例如在继续训练或验证时），则直接使用
        if self.label_map is not None and self.num_classes is not None:
            mapped_labels = np.array([self.label_map[l] for l in orig_labels], dtype=np.int32)
        else:
            # 生成标签映射
            unique_labels = sorted(list(set(orig_labels)))
            self.label_map = {orig: idx for idx, orig in enumerate(unique_labels)}  # 保存映射关系(原始标签: 映射标签(idx，即索引))
            mapped_labels = np.array([self.label_map[l] for l in orig_labels], dtype=np.int32)  # 映射后的标签           
            if len(self.label_map) > 3:
                sample_map = dict(list(self.label_map.items())[:3])
                print(f"Label mapping (original -> mapped): {sample_map} ... (共 {len(self.label_map)} 个条目)")
            else:
                print("Label mapping (original -> mapped):", self.label_map)
            # 获取类别数
            self.num_classes = len(unique_labels)

        # 保存类别信息
        self.classes = {f"class_{idx}": idx for idx in range(self.num_classes)}

        # 构建样本列表
        self.samples = [(subset, label) for subset, label in zip(subsets, mapped_labels)]

        # 打印最终加载信息
        print(f"Loaded {len(self.samples)} samples from '{split}' split across {len(self.classes)} classes.")

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

        # 如果不需要法向量，则只返回前三个坐标
        if not self.normal_channel:
            point_set = point_set[:, 0:3]

        return point_set, np.array([cls]).astype(np.int32)


if __name__ == '__main__':
    import torch
    import time

    # 测试数据加载器性能
    root_dir = '/share/home/202321008879/data/h5data/originbad'  # 手动修改
    split = 'test'  #   'train', 'test' 或 'eval' # 手动修改

    start_time = time.time()
    data = ModelNetDataLoader(root=root_dir, npoint=1200, split=split, uniform=False,
                              normal_channel=True, n_jobs=32)
    print(f"{split} samples: {len(data)}")
    end_time = time.time()
    print(f"Data loading completed in {end_time - start_time:.2f} seconds.")

    # 测试 DataLoader
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1000, shuffle=True, num_workers=16)
    for point, label in DataLoader:
        print(f"Batch points shape: {point.shape}, Batch labels shape: {label.shape}")
