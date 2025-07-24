# 这个版本2025.05.03，在非推理模式下（训练或验证），严格把 orig_labels 映射到 [0, num_classes-1] 的标签，以满足 F.nll_loss 的要求。
# 在推理模式下，如果某个 load1 的原始标签 l 存在于 label_map 中，则用 label_map[l]；否则就直接保留原始标签 l。
# 对应于所有颗粒的所有子集都保存到一个训练或测试h5文件的，同时还加入了标签
# 由于标签是从1开始，而训练脚本里计算损失函数loss = F.nll_loss(pred, target.long())要求标签都是从0开始，因此还是要进行映射
# 同时这个脚本还可以检测数据集的子集个数，以及每个子集的点云个数
# mapped_labels 用于为每个样本在数据加载过程中提供“已映射”的标签，这个标签既满足神经网络训练（或者推理时的某些计算）的要求，
# 同时也能在后续结果处理中利用 inverse_label_map 恢复原始标签或进行匹配。所以，mapped_labels 就是对 h5 中原始标签按指定规则进行预处理之后的结果。
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
    def __init__(self, root, npoint=1200, split='train', uniform=False, normal_channel=True, n_jobs=4, label_map=None, num_classes=None, infer=False):
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
            infer: Whether to use inference mode for label mapping.
        """
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.split = split
        self.normal_channel = normal_channel
        self.label_map = label_map
        self.num_classes = num_classes
        self.infer = infer

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
            if self.infer:
                # 推理阶段，多出来的标签其实不参与神经网络计算，所以随便给个标签
                print("推理阶段，使用提供的标签映射，并进行修改，以适应新的标签")
                print("修改前标签映射长度:", len(self.label_map))
                mapped_labels = np.array([self.label_map.get(l, l) for l in orig_labels], dtype=np.int32)
                print("修改后标签映射长度:", len(set(mapped_labels)))
            elif split == 'test':
                mapped_labels = np.array([self.label_map[l] for l in orig_labels], dtype=np.int32)
                print(f"测试集加载，使用训练集生成的标签映射，标签映射长度: {len(self.label_map)}")
            else:
                mapped_labels = np.array([self.label_map[l] for l in orig_labels], dtype=np.int32)
                print(f'继续训练或验证，使用提供的标签映射，标签映射长度: {len(self.label_map)}')
           
        else:
            # 生成标签映射
            print("训练新模型，自动生成标签映射")
            unique_labels = sorted(list(set(orig_labels)))
            self.label_map = {orig: idx for idx, orig in enumerate(unique_labels)}  # 保存映射关系(原始标签: 映射标签(idx，即索引))
            mapped_labels = np.array([self.label_map[l] for l in orig_labels], dtype=np.int32)  # 映射后的标签           
            if len(self.label_map) > 3:
                first_sample_map = dict(list(self.label_map.items())[:3])
                backup_sample_map = dict(list(self.label_map.items())[-3:])
                print(f"Label mapping (original -> mapped): {first_sample_map} ... {backup_sample_map} (共 {len(self.label_map)} 个条目)")
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

    ckpt = torch.load('experiment/originai_classes25333_points1200_2025-05-02_11-25/checkpoints/originai-0.999921-0100.pth', map_location='cpu')
    num_class = ckpt.get('num_classes', None)
    label_map = ckpt.get('label_map', None)
    # 测试数据加载器性能
    root_dir = '/share/home/202321008879/data/h5data/load1_ai'  # 手动修改
    split = 'eval'  #   'train', 'test' 或 'eval' # 手动修改

    start_time = time.time()
    # data = ModelNetDataLoader(root=root_dir, npoint=1200, split=split, uniform=False,
    #                           normal_channel=True, n_jobs=32, )
    data = ModelNetDataLoader(root=root_dir, npoint=1200, split=split, uniform=False,
                              normal_channel=True, label_map=label_map, num_classes=num_class, infer=True)
    print(f"{split} samples: {len(data)}")
    end_time = time.time()
    print(f"Data loading completed in {end_time - start_time:.2f} seconds.")

    # 测试 DataLoader
    DataLoader = torch.utils.data.DataLoader(data, batch_size=2000, shuffle=True, num_workers=16)
    for point, label in DataLoader:
        print(f"Batch points shape: {point.shape}, Batch labels shape: {label.shape}")
