# 这个版本不生成H5文件，而是显示fps这个方法每个子集的采样点数量情况
import os
import numpy as np
import open3d as o3d
from tifffile import imread
from skimage.measure import marching_cubes
from joblib import Parallel, delayed
import time
import random
import torch
from torch_cluster import fps
from numpy import pad


def get_bounding_box_for_label(data, label):
    """
    逐片扫描数据（沿第一维，即 x 轴），计算指定标签的最小包围盒，
    避免一次性生成整个布尔数组，降低内存占用。
    返回两个数组，分别为最小和最大坐标；如果找不到该标签，则抛出异常。
    """
    x_min, y_min, z_min = np.inf, np.inf, np.inf
    x_max, y_max, z_max = -np.inf, -np.inf, -np.inf
    # data的形状为 (x, y, z)
    for x in range(data.shape[0]):
        slice_data = data[x]  # 当前切片，形状 (y, z)
        indices = np.nonzero(slice_data == label)  # 返回两个一维数组：y 和 z 的索引
        if indices[0].size == 0:
            continue
        y_idx, z_idx = indices
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y_idx.min())
        y_max = max(y_max, y_idx.max())
        z_min = min(z_min, z_idx.min())
        z_max = max(z_max, z_idx.max())
    if x_min == np.inf:
        raise ValueError(f"Label {label} not found in data.")
    return np.array([x_min, y_min, z_min]), np.array([x_max, y_max, z_max])


def extract_surface_points_and_normals(data, label):
    """
    提取单个标签的表面点并计算法向量。
    """
    min_bound, max_bound = get_bounding_box_for_label(data, label)
    sub_volume = data[min_bound[0]:max_bound[0] + 1,
                      min_bound[1]:max_bound[1] + 1,
                      min_bound[2]:max_bound[2] + 1]
    binary_mask = (sub_volume == label).astype(np.bool_)
    # 对二值图像进行填充
    binary_mask = pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    verts, _, _, _ = marching_cubes(binary_mask, level=0)
    # 调整顶点坐标，去掉填充的影响
    verts -= 1  # 因为填充了 1 层，所以减去 1
    verts += min_bound  # 将局部坐标调整为全局坐标

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=15))
    pcd.orient_normals_consistent_tangent_plane(k=10)
    normals = np.asarray(pcd.normals)

    combined_data = np.concatenate([verts, normals], axis=1)  # (x, y, z, nx, ny, nz)
    return combined_data


def fps_torch(label, points, num_samples):
    """
    使用 torch_cluster 的 FPS 实现。
    Args:
        points: 输入点云 (N, 3) 的坐标。
        num_samples: 需要采样的点数。
    Returns:
        sampled_points: 采样后的点云坐标。
    """
    points = torch.tensor(points, dtype=torch.float32)  # 转为 Tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    points = points.to(device)

    batch = torch.zeros(points.size(0), dtype=torch.long, device=device)  # 单批次
    sampled_indices = fps(points, batch, ratio=num_samples / points.size(0))  # FPS 采样

    return points[sampled_indices].cpu().numpy()  # 转回 NumPy


def fps_sampling(label, points_with_normals, num_samples):
    """
    使用 torch_cluster 实现的 FPS。
    即使点云数量不足，也始终返回采样结果（可能会重复采样）。
    """
    points = points_with_normals[:, :3]
    sampled_points = fps_torch(label, points, num_samples)
    # 测试：看看还会不会出问题
    if sampled_points.shape[0] > num_samples:
        print(f"Label {label} has {sampled_points.shape[0]},"
              f" which will cause problem in number.")
        sampled_points = sampled_points[:num_samples]  # 多了就删
        print(f"Label {label} now has been revised and has {sampled_points.shape[0]},")
    elif sampled_points.shape[0] < num_samples:
        print(f"Label {label} has {sampled_points.shape[0]},"
              f" which will cause problem in number.")
        pad_count = num_samples - sampled_points.shape[0]
        pad_rows = np.repeat(sampled_points[-1:, :], pad_count, axis=0)
        sampled_points = np.concatenate([sampled_points, pad_rows], axis=0)  # 少了就重复
        print(f"Label {label} now has been revised and has {sampled_points.shape[0]},")
    # 将采样点的法向量加入
    sampled_indices = [
        np.where(np.all(points_with_normals[:, :3] == point, axis=1))[0][0] for point in sampled_points
    ]
    return points_with_normals[sampled_indices]


def process_label(data, label, num_subsets=30, num_points_per_subset=1200):
    """
    处理单个标签，生成 FPS 采样子集，并返回子集和标签。
    如果点云数量不足，则记录不足情况，但仍使用 FPS 采样生成子集（可能重复采样）。
    返回值增加一个布尔标志，表示是否点数不足。
    """
    start_time = time.time()

    points_with_normals = extract_surface_points_and_normals(data, label)

    insufficient = False
    if len(points_with_normals) < num_points_per_subset:
        print(f"Warning label {label}: not enough points ({len(points_with_normals)} < {num_points_per_subset}). Still generating subset.")
        insufficient = True
    # 以上代码是不跳过数量不足的颗粒，只是记录标签

    subsets = []
    for i in range(num_subsets):
        sampled_points = fps_sampling(label, points_with_normals, num_points_per_subset)
        actual_points = len(sampled_points)
        if actual_points < num_points_per_subset:
            print(f"Label {label} has {len(points_with_normals)}, which cause problem in number.")
            print(f"Label {label}, Subset {i + 1} has {actual_points} points (less than {num_points_per_subset}).")
        elif actual_points > num_points_per_subset:
            print(f"Label {label} has {len(points_with_normals)}, which cause problem in number.")
            print(f"Label {label}, Subset {i + 1} has {actual_points} points (more than {num_points_per_subset}).")

    elapsed_time = time.time() - start_time
    if label % 100 == 0:
        print(f"Processed label {label} in {elapsed_time:.2f} seconds.")

    return subsets, label, elapsed_time, insufficient


def process_tif_to_fps_subsets(tif_file, output_dir,
                               num_subsets=30, num_points_per_subset=1200,
                               train_ratio=0.85, n_jobs=4):
    """
    直接从 tif 文件生成最终的 FPS 采样子集，不保存为H5文件，只显示每个子集的采样点数量情况。
    """
    data = imread(tif_file)
    print(f"label数据的维数(z,y,x)：{data.shape}")
    # 读取时，z作为数组的第0轴了，y为第1轴，因此要进行纠正，把2轴(avizo里为x轴的数据)的数据转到第0轴
    data = np.transpose(data, (2, 1, 0)).copy()
    print(f"修改后数据的维数(x,y,z)：{data.shape}")
    labels = np.unique(data)
    labels = labels[labels > 0]

    print(f"Processing {len(labels)} labels...")



    def process_wrapper(label):
        return process_label(data, label, num_subsets, num_points_per_subset)

    results = Parallel(n_jobs=n_jobs)(delayed(process_wrapper)(label) for label in labels)




if __name__ == '__main__':
    tif_file = 'I:/avizo-data/origin_new.label.tif'
    output_dir = 'I:/sandprocess/data/originnew/'

    # tif_file = 'I:/sandprocess/data/originbad/originbad.label.tif'
    # output_dir = 'I:/sandprocess/data/originbad/test'

    # 直接生成最终的 FPS 子集，显示每个子集的采样点数量情况，同时记录并保存点云数量不足的标签
    process_tif_to_fps_subsets(tif_file, output_dir, num_subsets=30,
                               num_points_per_subset=1200, train_ratio=0.85, n_jobs=14)