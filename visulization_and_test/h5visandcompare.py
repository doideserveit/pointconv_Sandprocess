# 同时显示FPS子集点云和tif显示的原始颗粒
import os
import numpy as np
import open3d as o3d
import h5py
from tifffile import imread
from skimage.measure import marching_cubes
from numpy import pad

def load_fps_pointcloud(h5_file, label, subset_index=0, show_normals=False):
    """
    加载H5文件中的FPS点云数据
    
    参数:
        h5_file: H5文件路径
        label: 要可视化的标签
        subset_index: 子集索引
        show_normals: 是否显示法向量
    """
    with h5py.File(h5_file, 'r') as f:
        subsets = f['subsets'][:]
        h5_labels = f['labels'][:]
        print(f'subsets shape: {subsets.shape}')

    # 筛选出对应label的子集
    indices = np.where(h5_labels == label)[0]
    if len(indices) == 0:
        raise ValueError(f"未在 {h5_file} 中找到标签 {label} 的子集数据。")

    # 选择指定的子集
    points_with_normals = subsets[indices[subset_index]]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    points = points_with_normals[:, :3]
    normals = points_with_normals[:, 3:]
    
    pcd.points = o3d.utility.Vector3dVector(points)
    if show_normals:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    pcd.paint_uniform_color([1, 0, 0])  # FPS点云设为红色
    
    return pcd, show_normals

def visualize_marching_cubes_mesh(tif_file, label):
    """使用Marching Cubes算法生成网格显示颗粒"""
    # 加载并处理TIF数据
    data = imread(tif_file)
    data = np.transpose(data, (2, 1, 0)).copy()
    
    # 提取指定标签的表面
    binary_mask = (data == label).astype(np.bool_)
    binary_mask = pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    
    # 使用Marching Cubes提取表面网格
    verts, faces, _, _ = marching_cubes(binary_mask, level=0)
    verts -= 1  # 调整填充带来的偏移
    
    # 创建网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色网格
    mesh.compute_triangle_normals()
    
    return mesh

def visualize_voxel_pointcloud(tif_file, label):
    """将颗粒体素显示为点云"""
    # 加载并处理TIF数据
    data = imread(tif_file)
    data = np.transpose(data, (2, 1, 0)).copy()
    # 指定表面
    binary_mask = (data == label).astype(np.bool_)
    binary_mask = pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    verts, faces, _, _ = marching_cubes(binary_mask, level=0)
    verts -= 1  # 调整由于填充带来的偏移
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.paint_uniform_color([0, 0, 1])  # 蓝色点云
    
    return pcd

def visualize_volume_rendering(tif_file, label):
    """使用体绘制方法显示颗粒"""
    # 加载并处理TIF数据
    data = imread(tif_file)
    data = np.transpose(data, (2, 1, 0)).copy()
    
    # 创建体素网格
    voxel_size = 1.0
    voxel_grid = o3d.geometry.VoxelGrid.create_dense(
        origin=[0, 0, 0],
        color=[0, 1, 0],  # 绿色
        voxel_size=voxel_size,
        width=data.shape[0],
        height=data.shape[1],
        depth=data.shape[2]
    )
    
    # 填充属于目标标签的体素
    x, y, z = np.where(data == label)
    for xi, yi, zi in zip(x, y, z):
        voxel_grid.set_voxel([xi, yi, zi], [0, 1, 0])  # 绿色标记
    
    return voxel_grid

def visualize_particles(h5_file, tif_file, label, subset_index=0, 
                       visualization_methods=['fps', 'marching_cubes'],
                       show_fps_normals=False):
    """
    可视化颗粒的主函数，可选择多种可视化方法
    
    参数:
        h5_file: H5文件路径
        tif_file: TIF文件路径
        label: 要可视化的标签
        subset_index: FPS子集索引
        visualization_methods: 可视化方法列表，可选值:
            'fps' - 显示FPS点云
            'marching_cubes' - 显示Marching Cubes网格
            'voxel_cloud' - 显示体素点云
            'volume' - 显示体绘制
        show_fps_normals: 是否显示FPS点云的法向量
    """
    # 收集要显示的几何体
    geometries = []
    
    # 添加坐标轴
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])
    # geometries.append(mesh_frame)
    
    # 根据选择的方法添加几何体
    if 'fps' in visualization_methods:
        fps_pcd, show_normals = load_fps_pointcloud(
            h5_file, label, subset_index, show_fps_normals
        )
        geometries.append(fps_pcd)
    
    if 'marching_cubes' in visualization_methods:
        mc_mesh = visualize_marching_cubes_mesh(tif_file, label)
        geometries.append(mc_mesh)
    
    if 'voxel_cloud' in visualization_methods:
        voxel_pcd = visualize_voxel_pointcloud(tif_file, label)
        geometries.append(voxel_pcd)
    
    if 'volume' in visualization_methods:
        volume = visualize_volume_rendering(tif_file, label)
        geometries.append(volume)

    
    # 配置渲染选项
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"颗粒可视化 - 标签 {label}",
        point_show_normal=show_normals,
        mesh_show_wireframe=True, # 显示网格
        mesh_show_back_face=True, # 显示背面
    )

# 使用示例
if __name__ == "__main__":
    # 配置文件路径和参数
    label = 19000  # 要可视化的标签
    subset_index = 0  # FPS子集索引
    tif_file = "I:/avizo-data/load1_ai_label.tif"  # 替换为实际的TIF文件路径
    h5_file = "I:/data/h5data/load1_ai/all_test_fps_subsets.h5"  # 替换为实际的H5文件路径
    
    # 选择要使用的可视化方法，可自由组合
    # 可选方法: 'fps', 'marching_cubes', 'voxel_cloud', 'volume'
    visualization_methods = ['fps', 'marching_cubes']  # 同时显示FPS点云和Marching Cubes网格
    
    # 控制是否显示FPS点云的法向量
    show_normals = False
    
    # 执行可视化
    visualize_particles(
        h5_file, tif_file, label, subset_index, 
        visualization_methods, show_fps_normals=show_normals
    )