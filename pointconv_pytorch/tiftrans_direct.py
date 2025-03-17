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

    verts, _, _, _ = marching_cubes(binary_mask, level=0)

    verts += min_bound

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=15))
    pcd.orient_normals_consistent_tangent_plane(k=10)
    normals = np.asarray(pcd.normals)

    combined_data = np.concatenate([verts, normals], axis=1)  # (x, y, z, nx, ny, nz)
    return combined_data

def fps_torch(points, num_samples):
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
    """
    if len(points_with_normals) < num_samples:
        return None  # 如果点云数量不足，直接返回 None

    # 提取点坐标
    points = points_with_normals[:, :3]

    # 调用 PyTorch FPS 实现
    sampled_points = fps_torch(points, num_samples)

    # 将采样点的法向量加入
    sampled_indices = [
        np.where(np.all(points_with_normals[:, :3] == point, axis=1))[0][0] for point in sampled_points
    ]
    return points_with_normals[sampled_indices]


def process_label(data, label,  train_dir, test_dir,  train_ratio=0.85, num_subsets=30, num_points_per_subset=1200):
    """
    处理单个标签，生成 FPS 采样子集，并按 train/test 划分保存。
    """
    # 检查训练和测试文件是否已存在
    train_label_dir = os.path.join(train_dir, f"class_{label}")
    test_label_dir = os.path.join(test_dir, f"class_{label}")
    train_npz_filename = os.path.join(train_label_dir, f"{label}_train_fps_subsets.npz")
    test_npz_filename = os.path.join(test_label_dir, f"{label}_test_fps_subsets.npz")

    if os.path.exists(train_npz_filename) and os.path.exists(test_npz_filename):
        print(f"Label {label} already processed. Skipping...")
        return None, False, label  # 返回 None, False, label

    start_time = time.time()

    points_with_normals = extract_surface_points_and_normals(data, label)

    if len(points_with_normals) < num_points_per_subset:
        print(f"Skipping label {label}: not enough points ({len(points_with_normals)} < {num_points_per_subset})")
        return None, True, label  # 返回 None, True, label

    subsets = []
    for i in range(num_subsets):
        sampled_points = fps_sampling(label, points_with_normals, num_points_per_subset)
        subsets.append(sampled_points)
    # 检查生成的子集
    subsets = np.array(subsets, dtype=np.float32)  # 确保生成数据为 float32
    print(f"Label {label} - Generated {len(subsets)} subsets with shape: {subsets.shape}")

    # 随机划分子集到训练集和测试集
    random.shuffle(subsets)
    train_count = int(len(subsets) * train_ratio)
    train_subsets = subsets[:train_count]
    test_subsets = subsets[train_count:]

    # 保存训练集子集
    os.makedirs(train_label_dir, exist_ok=True)
    np.savez_compressed(train_npz_filename, subsets=train_subsets)

    # 保存测试集子集
    os.makedirs(test_label_dir, exist_ok=True)
    np.savez_compressed(test_npz_filename, subsets=test_subsets)

    elapsed_time = time.time() - start_time
    print(f"Processed label {label} in {elapsed_time:.2f} seconds.")

    return elapsed_time, False, label  # 返回 elapsed_time, False, label


def process_tif_to_fps_subsets(tif_file, output_dir,
                               num_subsets=30, num_points_per_subset=1200,
                               train_ratio=0.85, n_jobs=4):
    """
    直接从 tif 文件生成最终的 FPS 采样子集，并划分为训练集和测试集。
    """
    data = imread(tif_file)
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
    tif_file = 'I:/sandprocess/data/originbad/originbad.label.tif'
    output_dir = 'I:/sandprocess/data/originbad/fps_subsets'

    # 直接生成最终的 FPS 子集，并划分为训练集和测试集
    process_tif_to_fps_subsets(tif_file, output_dir, num_subsets=30,
                               num_points_per_subset=1200, train_ratio=0.85, n_jobs=14)

# 下面这个是open3d的farthest_point_down_sample实现的fps，但是这个函数应该不是真正的fps，总是取不够点。
# def fps_sampling(label, points_with_normals, num_samples):
#     """
#     对点云和法向量进行最远点采样 (FPS)。
#     """
#     print(f"Label {label}: points_with_normals has {len(points_with_normals)} points.")
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points_with_normals[:, :3])
#     sampled_indices = pcd.farthest_point_down_sample(num_samples).points
#
#     sampled_points_indices = [
#         np.where(np.linalg.norm(points_with_normals[:, :3] - point, axis=1) < 1e-6)[0][0]
#         for point in sampled_indices
#     ]
#
#     sampled_points = points_with_normals[sampled_points_indices]
#
#     print(f"Label {label}: FPS sampling returned {len(sampled_points)} points.")
#
#     return sampled_points

# 下面的配合自动化脚本的，暂时不管了
# import os
# import numpy as np
# import open3d as o3d
# from tifffile import imread
# from skimage.measure import marching_cubes
# from joblib import Parallel, delayed
# import time
# import random
# import argparse
#
#
# def get_bounding_box_for_label(data, label):
#     coords = np.argwhere(data == label)
#     min_bound = coords.min(axis=0)
#     max_bound = coords.max(axis=0)
#     return min_bound, max_bound
#
#
# def extract_surface_points_and_normals(data, label):
#     min_bound, max_bound = get_bounding_box_for_label(data, label)
#     sub_volume = data[min_bound[0]:max_bound[0] + 1, min_bound[1]:max_bound[1] + 1, min_bound[2]:max_bound[2] + 1]
#     binary_mask = (sub_volume == label).astype(np.bool_)
#
#     verts, _, _, _ = marching_cubes(binary_mask, level=0)
#     verts += min_bound
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(verts)
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=15))
#     pcd.orient_normals_consistent_tangent_plane(k=10)
#     normals = np.asarray(pcd.normals)
#
#     combined_data = np.concatenate([verts, normals], axis=1)  # (x, y, z, nx, ny, nz)
#     return combined_data
#
#
# def fps_sampling(points_with_normals, num_samples):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points_with_normals[:, :3])
#     sampled_indices = pcd.farthest_point_down_sample(num_samples).points
#     sampled_points_indices = [np.where((points_with_normals[:, :3] == point).all(axis=1))[0][0] for point in
#                               sampled_indices]
#     sampled_points = points_with_normals[sampled_points_indices]
#     return sampled_points
#
#
# def process_label(data, label, train_dir, test_dir, train_ratio=0.85, num_subsets=30, num_points_per_subset=1200):
#     """
#     Process each label to generate subsets and save to train/test directories.
#     """
#     start_time = time.time()
#
#     train_label_dir = os.path.join(train_dir, f"class_{label}")
#     test_label_dir = os.path.join(test_dir, f"class_{label}")
#     os.makedirs(train_label_dir, exist_ok=True)
#     os.makedirs(test_label_dir, exist_ok=True)
#
#     # Check if subsets already exist
#     train_npz_file = os.path.join(train_label_dir, f"{label}_train_fps_subsets.npz")
#     test_npz_file = os.path.join(test_label_dir, f"{label}_test_fps_subsets.npz")
#     if os.path.exists(train_npz_file) and os.path.exists(test_npz_file):
#         print(f"Label {label} subsets already exist, skipping...")
#         return
#
#     points_with_normals = extract_surface_points_and_normals(data, label)
#
#     if len(points_with_normals) < num_points_per_subset:
#         print(f"Skipping label {label}: not enough points ({len(points_with_normals)} < {num_points_per_subset})")
#         return
#
#     subsets = []
#     for i in range(num_subsets):
#         sampled_points = fps_sampling(points_with_normals, num_points_per_subset)
#         subsets.append(sampled_points)
#
#     # Split subsets into train/test
#     random.shuffle(subsets)
#     train_count = int(len(subsets) * train_ratio)
#     train_subsets = subsets[:train_count]
#     test_subsets = subsets[train_count:]
#
#     # Save to train/test directories
#     np.savez_compressed(train_npz_file, subsets=train_subsets)
#     np.savez_compressed(test_npz_file, subsets=test_subsets)
#
#     elapsed_time = time.time() - start_time
#     print(f"Processed label {label} in {elapsed_time:.2f} seconds.")
#
#
# def main(passed_args=None):
#     """
#     Main function to generate FPS subsets.
#     Supports both standalone execution and external script parameter passing.
#     """
#     parser = argparse.ArgumentParser(description="Generate FPS subsets from .tif file.")
#     parser.add_argument('--tif_file', type=str, default='I:/avizo-data/origin.label.tif',
#                         help='Path to the input .tif file')
#     parser.add_argument('--output_dir', type=str, default='I:/sandprocess/data/origin/fps_subsets',
#                         help='Output directory for train/test subsets')
#     parser.add_argument('--num_subsets', type=int, default=30, help='Number of subsets per label')
#     parser.add_argument('--num_points_per_subset', type=int, default=1200, help='Number of points per subset')
#     parser.add_argument('--train_ratio', type=float, default=0.85, help='Train/test split ratio')
#     parser.add_argument('--n_jobs', type=int, default=14, help='Number of parallel jobs')
#
#     # Parse arguments
#     if passed_args is None:
#         args = parser.parse_args()  # Standalone execution
#     else:
#         args = parser.parse_args(passed_args)  # External script execution
#
#     # Load data
#     data = imread(args.tif_file)
#     labels = np.unique(data)
#     labels = labels[labels > 0]
#
#     print(f"Processing {len(labels)} labels...")
#
#     # Create output directories
#     train_dir = os.path.join(args.output_dir, "train")
#     test_dir = os.path.join(args.output_dir, "test")
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)
#
#     # Process labels in parallel
#     Parallel(n_jobs=args.n_jobs)(
#         delayed(process_label)(data, label, train_dir, test_dir, args.train_ratio, args.num_subsets,
#                                args.num_points_per_subset)
#         for label in labels
#     )
#     print("Finished processing all labels.")
#
#
# if __name__ == "__main__":
#     main()

