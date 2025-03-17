import os
import numpy as np
import open3d as o3d
from tifffile import imread
from skimage.measure import marching_cubes
from pointconv_pytorch.joblib import Parallel, delayed
import gc

# 这个版本效率太低了，大数据时不好用
# 这个分别保存了（1）点云坐标、法向量，以及（2）FPS提取的颗粒子集
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
    sub_volume = data[min_bound[0]:max_bound[0] + 1, min_bound[1]:max_bound[1] + 1, min_bound[2]:max_bound[2] + 1]
    binary_mask = (sub_volume == label).astype(np.bool_)
    verts, _, _, _ = marching_cubes(binary_mask, level=0)
    verts += min_bound  # 将相对坐标转换回全局坐标

    # 计算法向量
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=15))
    pcd.orient_normals_consistent_tangent_plane(k=10)
    normals = np.asarray(pcd.normals)

    combined_data = np.concatenate([verts, normals], axis=1)  # (x, y, z, nx, ny, nz)
    return label, combined_data


def save_pointcloud_with_normals(points_with_normals, label, output_dir):
    """
    保存每个点包含法向量的点云数据为 .txt 文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{label}_surface_points_with_normals.txt")
    np.savetxt(filename, points_with_normals, fmt='%.6f')
    return filename


def extract_and_save_surface_points_with_normals(tif_path, output_dir, batch_size=100):
    """
    从 .tif 文件提取表面点和法向量，并保存为兼容 ModelNetDataLoader 的 .txt 文件格式。
    """
    data = imread(tif_path)
    labels = np.unique(data)
    labels = labels[labels > 0]  # 去掉背景值 0

    print(f"Extracting surface points and normals for {len(labels)} labels...")

    # # 使用 joblib 并行处理每个标签
    # results = Parallel(n_jobs=-1)(delayed(extract_surface_points_and_normals)(data, label) for label in labels)
    #
    # for label, points_with_normals in results:
    #     save_pointcloud_with_normals(points_with_normals, label, output_dir)

    # 按批处理，避免内存过载
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} with {len(batch_labels)} labels...")
        results = Parallel(n_jobs=12)(  # 限制并行任务数为12，避免占用过多内存
            delayed(extract_surface_points_and_normals)(data, label) for label in batch_labels
        )

        for label, points_with_normals in results:
            save_pointcloud_with_normals(points_with_normals, label, output_dir)

        # 手动清理内存
        del results
        gc.collect()

def load_pointcloud_with_normals(file_path):
    """
    从 .txt 文件加载带法向量的点云数据。
    """
    return np.loadtxt(file_path)


def fps_open3d(points_with_normals, num_samples):
    """
    使用 Open3D 进行最远点采样 (FPS)。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_with_normals[:, :3])  # 只使用坐标部分
    sampled_indices = pcd.farthest_point_down_sample(num_samples).points

    # 将采样后的索引转换为点云格式
    sampled_points_indices = [np.where((points_with_normals[:, :3] == point).all(axis=1))[0][0] for point in
                              sampled_indices]
    sampled_points = points_with_normals[sampled_points_indices]
    return sampled_points


def generate_fps_subsets_with_normals(input_dir, output_dir, num_subsets=30, num_points_per_subset=1200):
    """
    从保存的表面点和法向量文件生成子集（并行化处理），按标签分类存储到 fps_subsets 文件夹。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    def process_fps_for_files(txt_file):
        label = txt_file.split('_')[0]  # 将文件名按下划线分割，提取类名
        # 创建以标签命名的文件夹
        label_dir = os.path.join(output_dir, f"class_{label}")
        os.makedirs(label_dir, exist_ok=True)

        file_path = os.path.join(input_dir, txt_file)
        points_with_normals = load_pointcloud_with_normals(file_path)

        if len(points_with_normals) < num_points_per_subset:
            print(f"Warning: Label {label} has less than {num_points_per_subset} points, skipping FPS.")
            return []

        subsets = []

        for i in range(num_subsets):
            sampled_points = fps_open3d(points_with_normals, num_points_per_subset)  # 使用 Open3D 的 FPS 实现
            subset_filename = os.path.join(label_dir, f'{label}_subset_{i:02d}.txt')
            np.savetxt(subset_filename, sampled_points, fmt='%.6f')
            subsets.append(subset_filename)

        return subsets

    # 使用 joblib 并行处理每个文件的 FPS 子集生成
    all_subsets = Parallel(n_jobs=-1)(delayed(process_fps_for_files)(txt_file) for txt_file in txt_files)
    print(f"Generated {sum(len(subset) for subset in all_subsets)} subsets from {len(txt_files)} input files.")

    return all_subsets


def generate_train_test_txt(output_dir, train_ratio=0.85):
    """
    生成 sandmodel_train.txt 和 sandmodel_test.txt 文件，用于记录和测试子集路径。
    """

    subset_files = []
    for root, _, files in os.walk(output_dir):  # 遍历所有子文件夹，收集所有的 .txt 文件路径
        for file in files:
            if file.endswith('.txt'):
                subset_files.append(os.path.join(root, file))

    np.random.shuffle(subset_files)  # 随机打乱文件列表

    train_count = int(len(subset_files) * train_ratio)
    train_files = subset_files[:train_count]
    test_files = subset_files[train_count:]

    base_dir = os.path.dirname(output_dir)
    with open(os.path.join(base_dir, 'sandmodel_train.txt'), 'w') as train_txt:
        for f in train_files:
            relative_path = os.path.relpath(f, base_dir)  # 写入相对路径
            train_txt.write(f"{relative_path}\n")

    with open(os.path.join(base_dir, 'sandmodel_test.txt'), 'w') as test_txt:
        for f in train_files:
            relative_path = os.path.relpath(f, base_dir)  # 写入相对路径
            test_txt.write(f"{relative_path}\n")


    print(f"Generated sandmodel_train.txt with {len(train_files)} files "
          f"and sandmodel_test.txt  with {len(test_files)} files.")


def process_tif_to_pointcloud_and_subsets(tif_file, surface_output_dir, subsets_output_dir, train_ratio=0.85):
    """
    1. 提取表面点云和法向量，并保存。
    2. 使用 FPS 生成 30 个子集，每个子集 1200 个点。
    3. 生成 sandmodel_train.txt 和 sandmodel_test.txt 文件
    3. 生成 sandmodel_shape_names.txt 文件
    """
    print("Step 1: Extracting and saving surface points with normals...")
    extract_and_save_surface_points_with_normals(tif_file, surface_output_dir)

    print("Step 2: Generating FPS subsets...")
    generate_fps_subsets_with_normals(surface_output_dir, subsets_output_dir)

    print("Step 3:  Generating train and test txt files....")
    generate_train_test_txt(subsets_output_dir, train_ratio)

    # 生成 sandmodel_shape_names.txt
    print("Step 4: Generating shape names file...")
    class_names = [f"class_{label}" for label in np.unique(imread(tif_file)) if label > 0]
    with open(os.path.join(os.path.dirname(subsets_output_dir), 'sandmodel_shape_names.txt'), 'w') as shape_names_txt:
        for class_name in class_names:
            shape_names_txt.write(f"{class_name}\n")

    print("Generated sandmodel_shape_names.txt.")


if __name__ == '__main__':
    tif_file = 'I:/avizo-data/origin.label.tif'  # 替换为实际 .tif 文件路径
    surface_output_dir = 'I:/sandprocess/data/origin/surface_points_with_normals'  # 保存表面点云的目录
    subsets_output_dir = 'I:/sandprocess/data/origin/fps_subsets'  # 保存子集的目录

    process_tif_to_pointcloud_and_subsets(tif_file, surface_output_dir, subsets_output_dir, train_ratio=0.85)
