import os
import numpy as np
from tifffile import imread
from pointconv_pytorch.joblib import Parallel, delayed
from skimage.measure import marching_cubes

# 这个是单独提取点云坐标的
def get_bounding_box_for_label(data, label):
    """
    计算指定标签的最小包围盒
    """
    coords = np.argwhere(data == label)
    min_bound = coords.min(axis=0)
    max_bound = coords.max(axis=0)
    return min_bound, max_bound


def extract_surface_points(data, label):
    """
    提取单个标签的表面点。
    """
    min_bound, max_bound = get_bounding_box_for_label(data, label)
    sub_volume = data[min_bound[0]:max_bound[0] + 1, min_bound[1]:max_bound[1] + 1, min_bound[2]:max_bound[2] + 1]
    binary_mask = (sub_volume == label).astype(np.bool_)
    verts, _, _, _ = marching_cubes(binary_mask, level=0)
    # 将相对坐标转换回全局坐标
    verts += min_bound
    return label, verts


def save_pointcloud_to_txt(points, label, output_dir):
    """
    保存点云数据为 .txt 文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{label}_surface_points.txt')
    np.savetxt(filename, points, fmt='%.6f')
    print(f"Saved point cloud for label {label} with {len(points)} points to {filename}")


def generate_surface_points(tif_path, output_dir):
    """
    从 .tif 文件生成并保存表面点云。
    """
    data = imread(tif_path)
    labels = np.unique(data)
    labels = labels[labels > 0]  # 去掉背景值 0

    print(f"Starting surface point extraction for {len(labels)} labels...")
    results = Parallel(n_jobs=-1)(delayed(extract_surface_points)(data, label) for label in labels)

    for label, points in results:
        print(f"Processing label {label} with {len(points)} surface points...")
        save_pointcloud_to_txt(points, label, output_dir)


if __name__ == '__main__':
    tif_file = 'I:/sandprocess/data/origintest/origintest.label.tif'  # 替换为实际 .tif 文件路径
    output_dir = 'I:/sandprocess/data/origintest/surface_points'  # 保存表面点云的目录

    generate_surface_points(tif_file, output_dir)
