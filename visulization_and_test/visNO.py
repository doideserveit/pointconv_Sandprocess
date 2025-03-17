import numpy as np
import open3d as o3d
from tifffile import imread
from skimage.measure import marching_cubes

# 这个是用来学习方法的，分别显示法向量点云和原始tif图的点云
def visualize_pointcloud_with_normals(file_path):
    """
    从文件加载带法向量的点云数据并进行可视化。
    """
    # 从文件加载点云数据
    points_with_normals = np.loadtxt(file_path)

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()

    # 分离点和法向量
    points = points_with_normals[:, :3]
    normals = points_with_normals[:, 3:]

    # 设置点云的点和法向量
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # 可视化点云和法向量
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)


def visualize_3d_surface_from_tif(tif_path, label):
    """
    从 TIF 文件加载指定标签的表面并进行可视化。
    """
    # 加载 .tif 数据
    data = imread(tif_path)

    # 提取指定标签的表面
    binary_mask = (data == label).astype(np.bool_)

    # 使用 Marching Cubes 提取表面网格
    verts, faces, _, _ = marching_cubes(binary_mask, level=0)

    # 创建 Open3D 点云对象
    surface_pcd = o3d.geometry.PointCloud()

    # 设置点云的点
    surface_pcd.points = o3d.utility.Vector3dVector(verts)

    # 可视化原始表面点云
    o3d.visualization.draw_geometries([surface_pcd], point_show_normal=True)


def compare_fps_with_original_surface(fps_file, tif_file, label):
    """
    对比 FPS 提取的子集和原始颗粒的表面。
    """
    print(f"Visualizing FPS point cloud for label {label}...")
    visualize_pointcloud_with_normals(fps_file)

    print(f"Visualizing original surface for label {label}...")
    visualize_3d_surface_from_tif(tif_file, label)


# 使用示例
tif_file = 'I:/sandprocess/data/origintest/origintest.label.tif'  # 替换为实际的 .tif 文件路径
fps_file = 'I:/sandprocess/data/origintest/fps_subsets/1_subset_00.txt'  # 替换为实际的 FPS 子集文件路径
label = 1  # 替换为需要对比的标签

compare_fps_with_original_surface(fps_file, tif_file, label)
