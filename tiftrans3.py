import os
import numpy as np
import open3d as o3d
from tifffile import imread
from skimage.measure import marching_cubes

# 这个也是别的实现方法，用做备份，实际用separate的那个
def load_surface_points_marching_cubes(tif_path):
    """
    使用 marching cubes 提取表面点。
    Args:
        tif_path: 3D .tif 文件路径。
    Returns:
        surface_points_dict: 每个标签对应的表面点坐标 (x, y, z)。
    """
    data = imread(tif_path)
    labels = np.unique(data)
    labels = labels[labels > 0]  # 去掉背景值 0

    surface_points_dict = {}
    for label in labels:
        print(f"Extracting surface points for label {label}...")
        binary_mask = (data == label).astype(np.int32)
        verts, _, _, _ = marching_cubes(binary_mask, level=0)  # 提取表面点 (x, y, z)
        surface_points_dict[label] = verts  # 只返回表面顶点坐标

    return surface_points_dict


def compute_normals_open3d(points):
    """
    使用 Open3D 计算法向量。
    Args:
        points: 表面点云坐标 (x, y, z)。
    Returns:
        法向量矩阵 (nx, ny, nz)。
    """
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 使用 KD 树邻域查询计算法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    normals = np.asarray(pcd.normals)

    # 确保法向量方向一致
    pcd.orient_normals_consistent_tangent_plane(k=10)
    return normals


def fps(points, num_samples):
    """
    最远点采样 (FPS)。
    Args:
        points: 点云坐标 (x, y, z)。
        num_samples: 采样点数。
    Returns:
        采样后的点云。
    """
    sampled_indices = [np.random.randint(len(points))]
    while len(sampled_indices) < num_samples:
        remaining_points = points[sampled_indices]
        dist = np.linalg.norm(points[:, None] - remaining_points[None, :], axis=-1)
        farthest_point_index = np.argmax(np.min(dist, axis=1))
        sampled_indices.append(farthest_point_index)
    return points[sampled_indices]


def save_pointcloud_to_txt(pointcloud, label, output_dir, sample_idx):
    """
    保存点云数据为 .txt 文件，符合 ModelNet40 格式。
    Args:
        pointcloud: 包含 (x, y, z, nx, ny, nz) 特征的点云数据。
        label: 当前颗粒的标签。
        output_dir: 保存路径。
        sample_idx: 采样序号。
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{label}_sample_{sample_idx:02d}.txt')
    np.savetxt(filename, pointcloud, fmt='%.6f')


def generate_pointcloud_subsets(tif_path, output_dir, num_subsets=30, num_points_per_subset=1200):
    """
    从 .tif 文件生成点云子集并保存为 .txt 格式。
    Args:
        tif_path: 3D .tif 文件路径。
        output_dir: 输出目录。
        num_subsets: 每个颗粒生成的子集数量。
        num_points_per_subset: 每个子集的点数。
    """
    surface_points_dict = load_surface_points_marching_cubes(tif_path)

    for label, points in surface_points_dict.items():
        print(f"Processing label {label} with {len(points)} surface points...")
        if len(points) < num_points_per_subset:
            print(f"Warning: Label {label} has less than {num_points_per_subset} points.")
            continue

        normals = compute_normals_open3d(points)  # 使用 Open3D 计算法向量
        combined_data = np.concatenate([points, normals], axis=1)  # (x, y, z, nx, ny, nz)

        for i in range(num_subsets):
            sampled_points = fps(combined_data, num_points_per_subset)  # FPS 采样 1200 个点
            save_pointcloud_to_txt(sampled_points, label, output_dir, i)


if __name__ == '__main__':
    tif_file = 'I:/sandprocess/data/origintest/origintest.label.tif'  # 替换为实际 .tif 文件路径
    output_dir = 'I:/sandprocess/data/origintest/pointcloud_data'  # 保存点云子集的目录

    generate_pointcloud_subsets(tif_file, output_dir)
