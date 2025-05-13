# 这个版本更新于tiftrans_250304上.在其基础上增加了分批次处理的功能.每个批次处理并行1000个标签, 并单独保存每个批次的结果.
# 处理完所有标签后, 将所有批次的结果合并为一个h5文件.这样可以随时中断处理, 并在需要时继续处理剩余的标签.

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
import glob
import json

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


def get_batch_offset(output_dir, prefix):
    # 返回以prefix开头的批次文件数量
    existing = glob.glob(os.path.join(output_dir, f"{prefix}_*.npz"))
    return len(existing)


def save_batch_file(output_dir, prefix, batch_index, data_dict):
    batch_file = os.path.join(output_dir, f"{prefix}_{batch_index}.npz")
    np.savez_compressed(batch_file, **data_dict)
    print(f"Batch {batch_index} results saved to {batch_file}")


def merge_batches_to_h5(output_dir, batch_prefix, final_h5_filename, dataset_keys):
    """
    读取output_dir中以batch_prefix开头的npz文件，将指定数据（dataset_keys）
    合并后调用save_h5保存到final_h5_filename中。
    dataset_keys：例如["train_subsets", "train_labels"]
    """
    all_data = {key: [] for key in dataset_keys}
    batch_files = sorted(glob.glob(os.path.join(output_dir, f"{batch_prefix}_*.npz")),
                         key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for batch_file in batch_files:
        d = np.load(batch_file)
        for key in dataset_keys:
            all_data[key].append(d[key])
    for key in dataset_keys:
        if all_data[key]:
            all_data[key] = np.concatenate(all_data[key], axis=0)
        else:
            all_data[key] = np.array([])
    save_h5(final_h5_filename, all_data[dataset_keys[0]], all_data[dataset_keys[1]])
    print(f"Final H5 file saved to {final_h5_filename}")


def process_batches(data, labels_to_process, checkpoint_file, batch_prefix, process_label_args, n_jobs, batch_size):
    """
    对标签列表按照batch_size进行批处理，利用Parallel执行process_label，
    每批结束后yield处理结果、当前批次id及已有的标签列表（由调用者负责更新checkpoint）。
    process_label_args是传递给process_label的额外参数字典。
    """
    processed_labels = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_labels = json.load(f)
    batch_offset = get_batch_offset(output_dir, batch_prefix)
    for i in range(0, len(labels_to_process), batch_size):
        batch_labels = labels_to_process[i:i + batch_size]
        print(f"Processing batch {batch_offset + (i // batch_size + 1)}: labels from {batch_labels[0]} to {batch_labels[-1]}")
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process_label)(data, label, **process_label_args) for label in batch_labels
        )
        yield results, batch_offset + (i // batch_size + 1), processed_labels


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
                               train_ratio=0.85, n_jobs=4, batch_size=1000):
    data = imread(tif_file)
    print(f"label数据的维数(z,y,x)：{data.shape}")
    data = np.transpose(data, (2, 1, 0)).copy()
    print(f"修改后数据的维数(x,y,z)：{data.shape}")
    all_labels = np.unique(data)
    all_labels = all_labels[all_labels > 0]
    print(f"Processing {len(all_labels)} labels...")

    checkpoint_file = os.path.join(output_dir, "checkpoint_fps.json")
    processed_labels = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_labels = json.load(f)
    labels_to_process = [int(l) for l in all_labels if int(l) not in processed_labels]

    process_args = {"num_subsets": num_subsets, "num_points_per_subset": num_points_per_subset}
    for results, batch_index, proc_labels in process_batches(data, labels_to_process, checkpoint_file, "partial_batch",
                                                             process_args, n_jobs, batch_size):
        batch_train_subsets = []
        batch_test_subsets = []
        batch_train_labels = []
        batch_test_labels = []
        for subsets, label, elapsed_time, insufficient, samp_issue in results:
            random.shuffle(subsets)
            train_count = int(len(subsets) * train_ratio)
            train_subsets = subsets[:train_count]
            test_subsets = subsets[train_count:]
            batch_train_subsets.append(train_subsets)
            batch_test_subsets.append(test_subsets)
            batch_train_labels.extend([label] * len(train_subsets))
            batch_test_labels.extend([label] * len(test_subsets))
            proc_labels.append(int(label))
        save_batch_file(output_dir, "partial_batch", batch_index, {
            "train_subsets": np.concatenate(batch_train_subsets, axis=0),
            "test_subsets": np.concatenate(batch_test_subsets, axis=0),
            "train_labels": np.array(batch_train_labels, dtype=np.int32),
            "test_labels": np.array(batch_test_labels, dtype=np.int32)
        })
        with open(checkpoint_file, "w") as f:
            json.dump(proc_labels, f)

    merge_batches_to_h5(output_dir, "partial_batch", os.path.join(output_dir, 'all_train_fps_subsets.h5'),
                        ["train_subsets", "train_labels"])
    merge_batches_to_h5(output_dir, "partial_batch", os.path.join(output_dir, 'all_test_fps_subsets.h5'),
                        ["test_subsets", "test_labels"])


def process_tif_to_eval_fps_subsets(tif_file, output_dir,
                                    num_eval_subsets=10, num_points_per_subset=1200,
                                    n_jobs=4, batch_size=1000):
    data = imread(tif_file)
    print(f"原始数据维数(z,y,x)：{data.shape}")
    data = np.transpose(data, (2, 1, 0)).copy()
    print(f"调整后数据维数(x,y,z)：{data.shape}")
    all_labels = np.unique(data)
    all_labels = all_labels[all_labels > 0]
    print(f"Processing eval subsets for {len(all_labels)} labels...")

    checkpoint_file = os.path.join(output_dir, "checkpoint_eval.json")
    processed_labels = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_labels = json.load(f)
    labels_to_process = [int(l) for l in all_labels if int(l) not in processed_labels]

    process_args = {"num_subsets": num_eval_subsets, "num_points_per_subset": num_points_per_subset}
    for results, batch_index, proc_labels in process_batches(data, labels_to_process, checkpoint_file, "partial_eval_batch",
                                                             process_args, n_jobs, batch_size):
        batch_eval_subsets = []
        batch_eval_labels = []
        for subsets, label, elapsed_time, insufficient, samp_issue in results:
            batch_eval_subsets.append(subsets)
            batch_eval_labels.extend([label] * len(subsets))
            proc_labels.append(int(label))
        save_batch_file(output_dir, "partial_eval_batch", batch_index, {
            "eval_subsets": np.concatenate(batch_eval_subsets, axis=0),
            "eval_labels": np.array(batch_eval_labels, dtype=np.int32)
        })
        with open(checkpoint_file, "w") as f:
            json.dump(proc_labels, f)

    merge_batches_to_h5(output_dir, "partial_eval_batch", os.path.join(output_dir, 'all_eval_fps_subsets.h5'),
                        ["eval_subsets", "eval_labels"])


if __name__ == '__main__':
    tif_file = '/share/home/202321008879/data/sandlabel/load1_ai_label.tif'
    output_dir = '/share/home/202321008879/data/h5data/load1_ai'
    print(f'torch.cuda.is_available:{torch.cuda.is_available()}')

    # process_tif_to_fps_subsets(tif_file, output_dir, num_subsets=30,
    #                            num_points_per_subset=1200, train_ratio=0.85, n_jobs=8, batch_size=1000)

    process_tif_to_eval_fps_subsets(tif_file, output_dir, num_eval_subsets=10,
                                    num_points_per_subset=1200, n_jobs=8, batch_size=1000)

