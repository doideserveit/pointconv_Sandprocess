# 这个版本.预先分好训练和测试集，将所有的砂颗粒的生成的子集数据合成为一个h5文件
# 即训练是一个h5文件，测试又是另一个h5文件。
# 同时在子集中包含数据的标签，同时这个代码不跳过点云数量不足的颗粒，而是仍旧用fps算法生成子集，只是把他们记录下来

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

# 这个在线程14时跑了一万多个颗粒后会爆内存
# def get_bounding_box_for_label(data, label):
#     """
#     计算指定标签的最小包围盒
#     """
#     coords = np.argwhere(data == label)
#     min_bound = coords.min(axis=0)
#     max_bound = coords.max(axis=0)
#     return min_bound, max_bound

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
    使用 torch_cluster 实现的 FPS。返回固定点数并记录问题"
    即使点云数量不足，也始终返回采样结果（可能会重复采样）。
    """
    # 由于 torch_cluster 的 FPS 算法在点云数量不足时会重复采样，因此要跳过数量不足的颗粒
    # 但是这样会产生映射的问题，因此不再判断点云数量是否足够，即使不足也返回采样结果（可能重复采样）。

    points = points_with_normals[:, :3]
    sampled_points = fps_torch(label, points, num_samples)

    # 初始化问题类型
    issue_type = None
    original_num = sampled_points.shape[0]

    # 处理点数过多
    if original_num > num_samples:
        sampled_points = sampled_points[:num_samples]  # 截断
    # 处理点数不足
    elif original_num < num_samples:
        pad_count = num_samples - original_num
        pad_rows = np.repeat(sampled_points[-1:, :], pad_count, axis=0)
        sampled_points = np.concatenate([sampled_points, pad_rows], axis=0)  # 少了就重复

    # 返回问题类型和原始点数
    if original_num != num_samples:
        issue_type = ("excess" if original_num > num_samples else "insufficient")
    # 采样点法向量
    sampled_indices = [
        np.where(np.all(points_with_normals[:, :3] == point, axis=1))[0][0] for point in sampled_points
    ]
    sampled = points_with_normals[sampled_indices]  # 索引
    return sampled, issue_type, original_num


def save_h5(filename, subsets, labels):
    """
    保存数据到 .h5 格式，subsets 和 labels 为对应的数据和标签数组
    """
    with h5py.File(filename, 'w') as f:
        f.create_dataset('subsets', data=subsets, compression="gzip")
        f.create_dataset('labels', data=labels, compression="gzip")
    print(f"Saved data to {filename}")


def process_label(data, label, num_subsets=30, num_points_per_subset=1200):
    """
    处理单个标签，生成 FPS 采样子集，并返回子集和标签。返回新增采样问题记录
    如果点云数量不足，则记录不足情况，但仍使用 FPS 采样生成子集（可能重复采样）。
    返回值增加一个布尔标志，表示是否点数不足。
    """
    start_time = time.time()
    points_with_normals = extract_surface_points_and_normals(data, label)

    # 原始点数不足记录
    insufficient = len(points_with_normals) < num_points_per_subset
    if insufficient:
        print(f"Warning label {label}: insufficient points ({len(points_with_normals)} < {num_points_per_subset})"
              f". Still generating subset.")

    # 以上代码是不跳过数量不足的颗粒，只是记录标签
    sampling_issues = None
    subsets = []
    for i in range(num_subsets):
        sampled, issue_type, original_num = fps_sampling(label, points_with_normals, num_points_per_subset)
        subsets.append(sampled)

        # 记录首次出现的采样问题
        if issue_type and not sampling_issues:
            sampling_issues = {
                'label': label,
                'type': issue_type,
                'original_num': original_num,
                'required_num': num_points_per_subset
            }
            print(f"Label {label} first sampling issue: {issue_type} points "
                  f"(with {original_num} points but require {num_points_per_subset})")

    subsets = np.array(subsets, dtype=np.float32)
    elapsed_time = time.time() - start_time
    if label % 100 == 0:
        print(f"已完成第{label}号颗粒处理，用时{elapsed_time:.1f}s")

    return subsets, label, elapsed_time, insufficient, sampling_issues


def process_tif_to_fps_subsets(tif_file, output_dir,
                               num_subsets=30, num_points_per_subset=1200,
                               train_ratio=0.85, n_jobs=4):
    """
    直接从 tif 文件生成最终的 FPS 采样子集，并划分为训练集和测试集，
    其中训练数据保存在一个 h5 文件中，测试数据保存在另一个 h5 文件中，且都包含标签。
    同时记录点云数量不足的颗粒，将这些标签写入 txt 文件中。
    """
    insufficient_labels = []  # 原始点数不足的标签
    sampling_issues = []  # 采样过程产生问题的标签
    data = imread(tif_file)
    print(f"label数据的维数(z,y,x)：{data.shape}")
    # 读取时，z作为数组的第0轴了，y为第1轴，因此要进行纠正，把2轴(avizo里为x轴的数据)的数据转到第0轴
    data = np.transpose(data, (2, 1, 0)).copy()
    print(f"修改后数据的维数(x,y,z)：{data.shape}")
    labels = np.unique(data)
    labels = labels[labels > 0]

    print(f"Processing {len(labels)} labels...")

    all_train_subsets = []
    all_test_subsets = []
    all_train_labels = []
    all_test_labels = []
    insufficient_labels = []  # 用于记录点云不足的标签

    def process_wrapper(label):
        return process_label(data, label, num_subsets, num_points_per_subset)

    results = Parallel(n_jobs=n_jobs)(delayed(process_wrapper)(label) for label in labels)
    # results格式示例: (subsets, label, elapsed_time, insufficient)
    # results可能类似：[
    #     (np.array([...]), 1, 2.5, True),  # 第一个标签的子集、标签、处理时间和是否足够点的布尔值
    #     (np.array([...]), 2, 1.2, False),  # 第二个标签由于点云数量不足，子集为 None
    #     (np.array([...]), 3, 3.1,True)  # 第三个标签的子集、标签、处理时间和是否足够点的布尔值
    for subsets, label, elapsed_time, insufficient, samp_issue in results:
        if insufficient:
            insufficient_labels.append(label)
        if samp_issue:  # 记录采样问题
            sampling_issues.append(samp_issue)
        # 这里的subsets是results中的元组的第一个元素的第一项，即单个颗粒的所有点云子集的numpy数组
        # 随机划分子集到训练集和测试集
        random.shuffle(subsets)  # 打乱子集顺序
        train_count = int(len(subsets) * train_ratio)
        train_subsets = subsets[:train_count]
        test_subsets = subsets[train_count:]

        all_train_subsets.append(train_subsets)
        all_test_subsets.append(test_subsets)
        all_train_labels.extend([label] * len(train_subsets))  # 因为大家的类别都是一样的
        all_test_labels.extend([label] * len(test_subsets))

    # 合并所有类别的数据
    if all_train_subsets:
        all_train_subsets = np.concatenate(all_train_subsets, axis=0)
        all_train_labels = np.array(all_train_labels, dtype=np.int32)
    else:
        all_train_subsets = np.array([])
        all_train_labels = np.array([])

    if all_test_subsets:
        all_test_subsets = np.concatenate(all_test_subsets, axis=0)
        all_test_labels = np.array(all_test_labels, dtype=np.int32)
    else:
        all_test_subsets = np.array([])
        all_test_labels = np.array([])

    # 保存到两个不同的 h5 文件中
    train_filename = os.path.join(output_dir, 'all_train_fps_subsets.h5')
    test_filename = os.path.join(output_dir, 'all_test_fps_subsets.h5')
    save_h5(train_filename, all_train_subsets, all_train_labels)
    save_h5(test_filename, all_test_subsets, all_test_labels)

    # 保存问题记录到文件
    def save_issues(filename, issues):
        with open(filename, 'w') as f:
            f.write("Label\tIssueType\tOriginalPoints\tRequiredPoints\n")
            for issue in issues:
                f.write(f"{issue['label']}\t{issue['type']}\t"
                        f"{issue['original_num']}\t{issue['required_num']}\n")
    # 打印不足的标签，并保存到 txt 文件
    print(f"Insufficient points in {len(insufficient_labels)} labels: {insufficient_labels}")
    insufficient_labels_file = os.path.join(output_dir, "insufficient_labels.txt")
    with open(insufficient_labels_file, "w") as f:
        for lab in insufficient_labels:
            f.write(f"class_{lab}\n")
    print(f"Insufficient labels written to {insufficient_labels_file}")
    # 保存采样问题
    if sampling_issues:
        sampling_issue_file = os.path.join(output_dir, "sampling_issues.txt")
        save_issues(sampling_issue_file, sampling_issues)
        print(f"Sampling issues saved to {sampling_issue_file}")

def process_tif_to_eval_fps_subsets(tif_file, output_dir,
                                    num_eval_subsets=10, num_points_per_subset=1200,
                                    n_jobs=4):
    """
    从 tif 文件生成指定数量的 eval FPS 子集，并合并保存到 all_eval_fps_subsets.h5
    Args:
        tif_file: 输入 tif 文件路径
        output_dir: 输出目录
        num_eval_subsets: 每个 label 生成的 eval 子数量
        num_points_per_subset: 每个子集需要的点数
        n_jobs: 并行处理线程数
    """
    data = imread(tif_file)
    print(f"原始数据维数(z,y,x)：{data.shape}")
    # 数据转置，调整为 (x,y,z)
    data = np.transpose(data, (2, 1, 0)).copy()
    print(f"调整后数据维数(x,y,z)：{data.shape}")
    labels = np.unique(data)
    labels = labels[labels > 0]

    print(f"Processing eval subsets for {len(labels)} labels...")

    eval_subsets_list = []
    eval_labels_list = []

    def process_eval_wrapper(label):
        # 使用指定数量生成 eval 子集
        return process_label(data, label, num_subsets=num_eval_subsets, num_points_per_subset=num_points_per_subset)

    results = Parallel(n_jobs=n_jobs)(delayed(process_eval_wrapper)(label) for label in labels)

    for subsets, label, elapsed_time, insufficient, samp_issue in results:
        # 将当前 label 的生成子集中全部作为 eval 数据
        eval_subsets_list.append(subsets)
        eval_labels_list.extend([label] * len(subsets))
        print(f"Label {label} eval generated {len(subsets)} subsets, time {elapsed_time:.1f}s")

    if eval_subsets_list:
        all_eval_subsets = np.concatenate(eval_subsets_list, axis=0)
        all_eval_labels = np.array(eval_labels_list, dtype=np.int32)
    else:
        all_eval_subsets = np.array([])
        all_eval_labels = np.array([])

    eval_filename = os.path.join(output_dir, 'all_eval_fps_subsets.h5')
    save_h5(eval_filename, all_eval_subsets, all_eval_labels)
    print(f"Eval FPS subsets saved to {eval_filename}")

if __name__ == '__main__':
    tif_file = '/share/home/202321008879/data/sandlabel/Load1_selpar.label.tif'
    output_dir = '/share/home/202321008879/data/h5data/load1_selpar'
    print(f'torch.cuda.is_available:{torch.cuda.is_available()}')

    # 直接生成最终的 FPS 子集，并划分为训练集和测试集，同时记录并保存点云数量不足的标签
    # process_tif_to_fps_subsets(tif_file, output_dir, num_subsets=30,
    #                            num_points_per_subset=1200, train_ratio=0.85, n_jobs=14)

    # 调用新增的 eval 子集生成函数（可根据需要放开注释或调整参数）
    process_tif_to_eval_fps_subsets(tif_file, output_dir, num_eval_subsets=10,
                                    num_points_per_subset=1200, n_jobs=16)
