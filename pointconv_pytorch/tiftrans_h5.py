# 下一个版本是tiftrans_250304.py，这个版本是每个class自己一个文件夹，标签在读取数据时是从文件夹的名字里读取，比较麻烦，还得映射

import os
import numpy as np
import open3d as o3d
from tifffile import imread
from skimage.measure import marching_cubes
from joblib import Parallel, delayed
import time
import random
import torch
import h5py
from torch_cluster import fps
from numpy import pad


def get_bounding_box_for_label(data, label):
    """
    计算指定标签的最小包围盒
    """
    coords = np.argwhere(data == label)
    min_bound = coords.min(axis=0)
    max_bound = coords.max(axis=0)
    return min_bound, max_bound


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

    # # 检查是否重复采样了：打印实际采样的点数量和唯一点数量
    # sampled_points = points[sampled_indices].cpu().numpy()
    # print(f"Label{label}:FPS sampled {len(sampled_points)} points, unique points: {len(np.unique(sampled_points, axis=0))}")

    return points[sampled_indices].cpu().numpy()  # 转回 NumPy

def fps_sampling(label, points_with_normals, num_samples):
    """
    使用 torch_cluster 实现的 FPS。
    """
    # 由于torch.cluster的FPS算法在点云数量不足时会重复采样，因此要跳过数量不足的颗粒
    if len(points_with_normals) < num_samples:
        return None  # 如果点云数量不足，直接返回 None

    # 提取点坐标
    points = points_with_normals[:, :3]

    # 调用 PyTorch FPS 实现
    sampled_points = fps_torch(label, points, num_samples)

    # 将采样点的法向量加入
    sampled_indices = [
        np.where(np.all(points_with_normals[:, :3] == point, axis=1))[0][0] for point in sampled_points
    ]
    return points_with_normals[sampled_indices]

def save_h5(filename, train_subsets, test_subsets):
    """
    保存数据到 .h5 格式
    """
    with h5py.File(filename, 'w') as f:
        if train_subsets is not None:
            f.create_dataset('train_subsets', data=train_subsets, compression="gzip")
        if test_subsets is not None:
            f.create_dataset('test_subsets', data=test_subsets, compression="gzip")
    # print(f"Saved data to {filename}")

def process_label(data, label, train_dir, test_dir, train_ratio=0.85, num_subsets=30, num_points_per_subset=1200):
    """
    处理单个标签，生成 FPS 采样子集，并按 train/test 划分保存。
    """
    # 创建训练和测试目录
    train_label_dir = os.path.join(train_dir, f"class_{label}")
    test_label_dir = os.path.join(test_dir, f"class_{label}")
    train_h5_filename = os.path.join(train_label_dir, f"{label}_train_fps_subsets.h5")
    test_h5_filename = os.path.join(test_label_dir, f"{label}_test_fps_subsets.h5")

    # 如果已存在，跳过
    if os.path.exists(train_h5_filename) and os.path.exists(test_h5_filename):
        print(f"Label {label} already processed. Skipping...")
        return None, False, label

    start_time = time.time()

    points_with_normals = extract_surface_points_and_normals(data, label)

    if len(points_with_normals) < num_points_per_subset:
        print(f"Skipping label {label}: not enough points ({len(points_with_normals)} < {num_points_per_subset})")
        return None, True, label
    print(f"class_{label} has {len(points_with_normals)} points.")
    subsets = []
    for i in range(num_subsets):
        sampled_points = fps_sampling(label, points_with_normals, num_points_per_subset)
        subsets.append(sampled_points)
    subsets = np.array(subsets, dtype=np.float32)

    # 随机划分子集到训练集和测试集
    random.shuffle(subsets)
    train_count = int(len(subsets) * train_ratio)
    train_subsets = subsets[:train_count]
    test_subsets = subsets[train_count:]

    # 保存数据到 h5 文件
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    save_h5(train_h5_filename, train_subsets, None)  # 只保存训练集
    save_h5(test_h5_filename, None, test_subsets)  # 只保存测试集

    elapsed_time = time.time() - start_time
    print(f"Processed label {label} in {elapsed_time:.2f} seconds.")

    return elapsed_time, False, label


def process_tif_to_fps_subsets(tif_file, output_dir,
                               num_subsets=30, num_points_per_subset=1200,
                               train_ratio=0.85, n_jobs=4):
    """
    直接从 tif 文件生成最终的 FPS 采样子集，并划分为训练集和测试集。
    """
    data = imread(tif_file)
    print(f"label数据的维数(z,y,x)：{data.shape}")
    # 读取时，z作为数组的第0轴了，y为第1轴，因此要进行纠正，把2轴(avizo里为x轴的数据)的数据转到第0轴
    data = np.transpose(data, (2, 1, 0)).copy()
    print(f"修改后数据的维数(x,y,z)：{data.shape}")
    labels = np.unique(data)
    labels = labels[labels > 0]

    print(f"Processing {len(labels)} labels...")

    skipped_labels = []  # 存储跳过的标签
    times = []  # 存储每个标签的处理时间

    # 创建训练和测试目录
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def process_wrapper(label):
        result = process_label(data, label, train_dir, test_dir,
                               train_ratio, num_subsets, num_points_per_subset)
        return result

    results = Parallel(n_jobs=n_jobs)(delayed(process_wrapper)(label) for label in labels)

    # 过滤掉 None 值并更新 skipped_labels 和 times
    for result in results:
        if result is not None:
            elapsed_time, skipped, label = result
            if skipped:
                skipped_labels.append(label)
            elif elapsed_time is not None:
                times.append(elapsed_time)

    # 打印汇总结果
    print(f"Finished processing {len(labels) - len(skipped_labels)} labels.")
    print(f"Skipped {len(skipped_labels)} labels due to insufficient points: {skipped_labels}")
    if times:
        print(f"Average processing time per label: {np.mean(times):.2f} seconds.")
    else:
        print("No labels were processed successfully.")

    # 保存跳过的标签到文件
    skipped_labels_file = os.path.join(output_dir, "skipped_labels.txt")
    with open(skipped_labels_file, "w") as f:
        for label in skipped_labels:
            f.write(f"class_{label}\n")
    print(f"Skipped labels written to {skipped_labels_file}")


if __name__ == '__main__':
    tif_file = 'I:/avizo-data/origin_new.label.tif'
    output_dir = 'I:/sandprocess/data/originnew/fps_subsets'

    # 直接生成最终的 FPS 子集，并划分为训练集和测试集
    process_tif_to_fps_subsets(tif_file, output_dir, num_subsets=30,
                               num_points_per_subset=1200, train_ratio=0.85, n_jobs=14)
