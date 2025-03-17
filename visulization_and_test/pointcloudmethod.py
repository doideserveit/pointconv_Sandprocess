
# 方法 1：使用 skimage.measure.marching_cubes 提取表面网格点
from skimage.measure import marching_cubes

def extract_surface_points_marching_cubes(tif_path):
    """
    使用 marching_cubes 算法提取颗粒的表面点云。
    Args:
        tif_path: 3D .tif 文件路径。
    Returns:
        surface_points_dict: 每个标签对应的表面点云坐标。
    """
    data = imread(tif_path)
    labels = np.unique(data)
    labels = labels[labels > 0]

    surface_points_dict = {}
    for label in labels:
        binary_data = (data == label).astype(np.uint8)
        verts, _, _, _ = marching_cubes(binary_data, level=0)
        surface_points_dict[label] = verts

    return surface_points_dict

# 方法 2：使用 PyntCloud 提取点云并计算法向量

from pyntcloud import PyntCloud
import pandas as pd

def extract_surface_points_with_normals(tif_path):
    """
    使用 PyntCloud 提取表面点并计算法向量。
    """
    data = imread(tif_path)
    labels = np.unique(data)
    labels = labels[labels > 0]

    surface_points_dict = {}
    for label in labels:
        coords = np.argwhere(data == label)
        cloud = PyntCloud(pd.DataFrame(coords, columns=['x', 'y', 'z']))
        cloud.add_scalar_field("normals", k_neighbors=10)
        surface_points = cloud.points[['x', 'y', 'z', 'nx', 'ny', 'nz']].values
        surface_points_dict[label] = surface_points

    return surface_points_dict

# 方法 3：使用 open3d 提取表面点
import open3d as o3d

def extract_surface_points_open3d(tif_path):
    """
    使用 open3d 提取颗粒的表面点。
    """
    data = imread(tif_path)
    labels = np.unique(data)
    labels = labels[labels > 0]

    surface_points_dict = {}
    for label in labels:
        coords = np.argwhere(data == label)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(coords)
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
        surface_points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)
        surface_points_dict[label] = np.hstack((surface_points, normals))

    return surface_points_dict
