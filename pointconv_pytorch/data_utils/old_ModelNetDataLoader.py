# import numpy as np
# import warnings
# import os
# from torch.utils.data import Dataset
# warnings.filterwarnings('ignore')
#
# def pc_normalize(pc):
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#     pc = pc / m
#     return pc
#
# def farthest_point_sample(point, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [N, D]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [npoint, D]
#     """
#     N, D = point.shape
#     xyz = point[:,:3]
#     centroids = np.zeros((npoint,))
#     distance = np.ones((N,)) * 1e10
#     farthest = np.random.randint(0, N)
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = xyz[farthest, :]
#         dist = np.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance, -1)
#     point = point[centroids.astype(np.int32)]
#     return point
#
# class ModelNetDataLoader(Dataset):
#     def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
#         """
#                Initialize the dataset loader.
#                Args:
#                    root: Root directory of the dataset.
#                    npoint: Number of points to sample.
#                    split: Dataset split ('train' or 'test').
#                    uniform: Whether to use farthest point sampling.
#                    normal_channel: Whether to include normal channels.
#                    cache_size: Size of the in-memory cache.
#                """
#         self.root = root
#         self.npoints = npoint
#         self.uniform = uniform
#         self.split = split
#         self.catfile = os.path.join(self.root, 'sandmodel_shape_names.txt')  # 根据自己的生成文本进行修改
#
#         self.cat = [line.rstrip() for line in open(self.catfile)]
#         self.classes = dict(zip(self.cat, range(len(self.cat))))
#         self.normal_channel = normal_channel
#
#         shape_ids = {}
#         shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'sandmodel_train.txt'))]
#         shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'sandmodel_test.txt'))]
#
#         assert (split == 'train' or split == 'test')
#         shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
#         # list of (shape_name, shape_txt_file_path) tuple
#         self.datapath = [(os.path.basename(os.path.dirname(shape_ids[split][i])),  # 类名
#                           os.path.join(self.root, shape_ids[split][i]))  # 完整路径
#                          for i in range(len(shape_ids[split]))]
#         print('The size of %s data is %d'%(split,len(self.datapath)))
#
#         self.cache_size = cache_size  # how many data points to cache in memory
#         self.cache = {}  # from index to (point_set, cls) tuple
#
#     def __len__(self):
#         return len(self.datapath)
#
#     def _get_item(self, index):
#         if index in self.cache:
#             point_set, cls = self.cache[index]
#         else:
#             filepath = self.datapath[index][1]
#             class_name = self.datapath[index][0] # Extract class name from file path
#             cls = self.classes[class_name]
#             cls = np.array([cls]).astype(np.int32)
#             point_set = np.loadtxt(filepath, delimiter=',').astype(np.float32)
#
#             if len(self.cache) < self.cache_size:
#                 self.cache[index] = (point_set, cls)
#
#         if self.uniform:
#             point_set = farthest_point_sample(point_set, self.npoints)
#         else:
#             if self.split == 'train':
#                 train_idx = np.array(range(point_set.shape[0]))
#                 point_set = point_set[train_idx[:self.npoints],:]
#             else:
#                 point_set = point_set[0:self.npoints,:]
#
#         point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
#
#         if not self.normal_channel:
#             point_set = point_set[:, 0:3]
#
#         return point_set, cls
#
#     def __getitem__(self, index):
#         return self._get_item(index)
#
#
# if __name__ == '__main__':
#     import torch
#
#     data = ModelNetDataLoader('I:\sandprocess\data\origintest',split='train', uniform=False, normal_channel=True,)
#     DataLoader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
#     for point,label in DataLoader:
#         print(point.shape)
#         print(label.shape)
#
#

import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
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
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        """
        Initialize the dataset loader.
        Args:
            root: Root directory of the dataset.
            npoint: Number of points to sample.
            split: Dataset split ('train' or 'test').
            uniform: Whether to use farthest point sampling.
            normal_channel: Whether to include normal channels.
            cache_size: Size of the in-memory cache.
        """
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.split = split
        self.catfile = os.path.join(self.root, 'sandmodel_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'sandmodel_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'sandmodel_test.txt'))]

        assert split in ['train', 'test']
        self.datapath = [(os.path.basename(os.path.dirname(shape_ids[split][i])),  # 类名
                          os.path.join(self.root, shape_ids[split][i]))  # 完整路径
                         for i in range(len(shape_ids[split]))]

        print(f'The size of {split} data is {len(self.datapath)}')

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index][1]
        else:
            filepath = self.datapath[index][1]  # 完整路径
            class_name = self.datapath[index][0]  # 类名
            cls = self.classes[class_name]
            cls = np.array([cls]).astype(np.int32)
            # point_set = np.loadtxt(filepath, delimiter=',').astype(np.float32)  # 我的数据不是用逗号分隔的
            point_set = np.loadtxt(filepath).astype(np.float32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.normal_channel:
            point_set = point_set[:, 0:3]

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('I:/sandprocess/data/origintest', split='train', uniform=False, normal_channel=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
