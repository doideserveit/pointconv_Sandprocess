# 这个脚本用来显示整个样本，不过计算速度有点慢
# 坐标轴红绿蓝分别是x,y,z
import numpy as np
import open3d as o3d
from tifffile import imread
from skimage.measure import marching_cubes
import skimage.transform as transform


def load_and_preprocess_tif(tif_path, downsample_factor=2):
    """
    读取 TIF 图像并调整轴顺序，并可选进行下采样以降低计算量。
    """
    data = imread(tif_path).astype(np.uint8)  # 读取并转换为 uint8 以降低内存使用
    data = np.transpose(data, (2, 1, 0)).copy()  # 调整轴顺序

    if downsample_factor > 1:
        data = transform.resize(data,
                                (data.shape[0] // downsample_factor,
                                 data.shape[1] // downsample_factor,
                                 data.shape[2] // downsample_factor),
                                anti_aliasing=True, mode='constant', preserve_range=True)
        data = data.astype(np.uint8)  # 重新转换数据类型

    return data


def extract_surface_mesh(data, step_size=2):
    """
    使用 Marching Cubes 提取表面网格，并计算法向量。
    """
    verts, faces, normals, _ = marching_cubes(data > 0, level=0, step_size=step_size)  # 计算法向量

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)  # 设置法向量

    # **改进渲染效果**
    mesh.paint_uniform_color([0.6, 0.6, 0.6])  # 统一灰色，避免花屏

    return mesh

def create_coordinate_frame(size=10.0):
    """
    创建坐标轴。
    """
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    return [coord_frame,]

def visualize_tif_sample(tif_path, downsample_factor=2, step_size=2):
    """
    读取 TIF 文件并快速显示完整的三轴颗粒样本，优化渲染效果。
    """
    print("Loading TIF file...")
    data = load_and_preprocess_tif(tif_path, downsample_factor)

    print("Extracting surface mesh...")
    mesh = extract_surface_mesh(data, step_size)

    print("Computing normals for better rendering...")
    mesh.compute_vertex_normals()  # 确保法向量正确计算

    print("Visualizing...")
    coord_objects = create_coordinate_frame(size=20.0)  # 添加更大的坐标轴和标签
    o3d.visualization.draw_geometries(
        [mesh] + coord_objects, mesh_show_back_face=True, mesh_show_wireframe=False, window_name="Triaxial Sample"
    )


# **示例使用**
if __name__ == "__main__":
    tif_file = "I:/avizo-data/origin_new.label.tif"  # 替换为实际路径
    visualize_tif_sample(tif_file, downsample_factor=2, step_size=2)  # 适当减少数据量，提高效率
