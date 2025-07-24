import os
import numpy as np
import open3d as o3d

#这个是单独提取法向量的
def compute_normals_open3d(points):
    """
    使用 Open3D 计算法向量。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 使用邻域参数
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=15))
    pcd.orient_normals_consistent_tangent_plane(k=10)

    normals = np.asarray(pcd.normals)
    return normals


def load_pointcloud_from_txt(file_path):
    """
    从 .txt 文件加载点云数据。
    """
    points = np.loadtxt(file_path)
    return points


def save_pointcloud_with_normals_to_txt(points_with_normals, label, output_dir):
    """
    保存含法向量的点云数据为 .txt 文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{label}_surface_points_with_normals.txt')
    np.savetxt(filename, points_with_normals, fmt='%.6f')
    print(f"Saved point cloud with normals for label {label} to {filename}")


def compute_and_save_normals_for_all_labels(input_dir, output_dir):
    """
    计算每个 .txt 文件的法向量并保存。
    """
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    for txt_file in txt_files:
        label = txt_file.split('_')[0]  # 获取标签
        file_path = os.path.join(input_dir, txt_file)
        print(f"Computing normals for {label}...")

        points = load_pointcloud_from_txt(file_path)
        normals = compute_normals_open3d(points)
        points_with_normals = np.concatenate([points, normals], axis=1)  # (x, y, z, nx, ny, nz)

        save_pointcloud_with_normals_to_txt(points_with_normals, label, output_dir)


if __name__ == '__main__':
    input_dir = 'I:/sandprocess/data/origintest/surface_points'  # 保存纯点云的目录
    output_dir = 'I:/sandprocess/data/origintest/surface_points_with_normals'  # 保存含法向量的点云目录

    compute_and_save_normals_for_all_labels(input_dir, output_dir)
