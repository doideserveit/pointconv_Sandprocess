# 这个代码用于检测生成的h5数据，将其可视化，并与tif文件直接生成的可视化进行对比

import os
import numpy as np
import open3d as o3d
import h5py
from tifffile import imread
from skimage.measure import marching_cubes
from numpy import pad

def visualize_fps_and_surface(h5_file, tif_file, label):
    """
    可视化 h5 文件中 FPS 提取的子集点云与 TIF 文件中提取的原始颗粒表面。
    h5 文件应包含 'subsets' 和 'labels' 数据集。

    Args:
        h5_file (str): h5 文件路径（训练集或测试集文件）
        tif_file (str): tif 文件路径
        label (int): 需要对比的标签
    """
    # 从 h5 文件中加载 FPS 提取的子集及其标签
    with h5py.File(h5_file, 'r') as f:
        subsets = f['subsets'][:]   # shape: (total_subsets, num_points, 6)
        print(f'len(subsets)={len(subsets)}')
        h5_labels = f['labels'][:]    # shape: (total_subsets,)
        print(f'len(h5_labels)={len(h5_labels)}')

    # 筛选出对应 label 的子集
    indices = np.where(h5_labels == label)[0]
    if len(indices) == 0:
        raise ValueError(f"未在 {h5_file} 中找到标签 {label} 的子集数据。")

    # 选择第一个符合条件的子集进行可视化
    points_with_normals = subsets[indices[24]]

    # 创建 Open3D 点云对象
    fps_pcd = o3d.geometry.PointCloud()

    # 分离点和法向量
    points = points_with_normals[:, :3]
    normals = points_with_normals[:, 3:]
    # 设置 FPS 点云的点和法向量
    fps_pcd.points = o3d.utility.Vector3dVector(points)
    fps_pcd.normals = o3d.utility.Vector3dVector(normals)

    # 创建坐标轴（可选）
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])

    # 法一：网格显示颗粒
    # 加载原始 TIF 数据
    data = imread(tif_file)
    data = np.transpose(data, (2, 1, 0)).copy()
    # 提取指定标签的表面
    binary_mask = (data == label).astype(np.bool_)
    # 使用 Marching Cubes 提取表面网格
    binary_mask = pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    verts, faces, _, _ = marching_cubes(binary_mask, level=0)
    verts -= 1  # 调整由于填充带来的偏移

    # 创建 Open3D 网格对象
    surface_mesh = o3d.geometry.TriangleMesh()
    # 设置网格的顶点和面
    surface_mesh.vertices = o3d.utility.Vector3dVector(verts)
    surface_mesh.triangles = o3d.utility.Vector3iVector(faces)
    # 设置网格的颜色
    surface_mesh.compute_vertex_normals()
    surface_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 设置表面为灰色

    # 可视化 FPS 提取的点云和原始表面网格
    o3d.visualization.draw_geometries(
        [fps_pcd, surface_mesh],
        point_show_normal=True,  # 显示法向量
        mesh_show_wireframe=True,  # 显示网格线框
        mesh_show_back_face=True,  # 显示反向面
        window_name="FPS and Original Surface"
    )
    # # 法二：下面的代码是点云来显示颗粒
    # data = imread(tif_file)
    # data = np.transpose(data, (2, 1, 0)).copy()
    # # 提取指定标签的表面
    # binary_mask = (data == label).astype(np.bool_)
    # # 对二值图像进行填充
    # binary_mask = pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    # # 使用 Marching Cubes 提取表面网格
    # verts, faces, _, _ = marching_cubes(binary_mask, level=0)
    # # 调整顶点坐标，去掉填充的影响
    # verts -= 1  # 因为填充了 1 层，所以减去 1
    #
    # # 创建 Open3D 点云对象
    # surface_pcd = o3d.geometry.PointCloud()
    #
    # # 设置点云的点
    # surface_pcd.points = o3d.utility.Vector3dVector(verts)
    #
    # # 可视化原始表面点云
    # o3d.visualization.draw_geometries([fps_pcd, surface_pcd, ], point_show_normal=True,
    #                                      window_name="FPS and Original Surface")

# 使用示例
label = 3  # 替换为需要对比的标签
tif_file = 'I:/sandprocess/data/originbad/originbad.label.tif'  # 替换为实际的 .tif 文件路径
h5_file = f'I:/sandprocess/data/originbad/test/all_train_fps_subsets.h5'  # 替换为实际的 h5 文件路径

visualize_fps_and_surface(h5_file, tif_file, label)
