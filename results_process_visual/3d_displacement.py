import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile as tiff
import os
import argparse
import matplotlib as mpl
from matplotlib import cm
from skimage.measure import find_contours

def load_properties(npz_path):
    """
    加载颗粒属性文件，返回 {label: centroid(index)}
    """
    data = np.load(npz_path, allow_pickle=True)
    labels = data['label']
    centroids = data['centroid(index)']  # 使用像素索引
    mapping = {}
    for i, lab in enumerate(labels):
        mapping[int(lab)] = np.array(centroids[i])
    return mapping

def compute_displacement_mapping_from_results_and_properties(results_npz_file, def_properties_npz, ref_properties_npz):
    """
    从results_npz_file中获取def label和其对应的ref label，
    然后分别从def/ref的properties中获取质心，计算位移（def质心 - ref质心）。
    返回: {def_label: displacement (3,)}，位移基于 corrected_centroid(physical) (mm)
    """
    def_mapping = load_properties(def_properties_npz)
    ref_mapping = load_properties(ref_properties_npz)
    data = np.load(results_npz_file, allow_pickle=True)
    results = data['results'].item()
    displacement_mapping = {}
    def_data = np.load(def_properties_npz, allow_pickle=True)
    ref_data = np.load(ref_properties_npz, allow_pickle=True)
    def_physical = dict(zip(def_data['label'], def_data['corrected_centroid(physical)']))
    ref_physical = dict(zip(ref_data['label'], ref_data['corrected_centroid(physical)']))
    
    for def_label, info in results.items():
        ref_label = info.get('predicted_reference_label', None)
        if ref_label is None:
            print(f"Warning: def_label {def_label} has no 'predicted_reference_label' in results, skipped.")
            continue
        def_label = int(def_label)
        ref_label = int(ref_label)
        if def_label not in def_mapping or ref_label not in ref_mapping:
            print(f"Warning: Label {def_label} or {ref_label} not found in properties, skipped.")
            continue
        def_centroid = np.array(def_physical[def_label])  # 使用物理坐标 (mm)
        ref_centroid = np.array(ref_physical[ref_label])  # 使用物理坐标 (mm)
        disp = def_centroid - ref_centroid
        if disp.shape != (3,):
            print(f"Warning: Label {def_label} displacement shape {disp.shape} is not (3,). Skipped.")
            continue
        displacement_mapping[def_label] = disp
    if len(displacement_mapping) == 0:
        print("Error: No valid displacement mapping found. Please check your results file and properties files.")
    return displacement_mapping

def get_max_displacement(displacement_mapping):
    """
    计算最大位移模长（不归一化）
    """
    return max(np.linalg.norm(v) for v in displacement_mapping.values())

def get_particle_contours(labels_3d, label, max_slices=2):
    """
    提取颗粒的 3D 边界（优化性能，限制切片数）
    返回：边界点列表 [(x, y, z), ...]
    """
    contours = []
    y_indices = np.linspace(0, labels_3d.shape[1] - 1, max_slices, dtype=int)
    for y in y_indices:
        slice_img = labels_3d[:, y, :]  # XZ 切片
        contours_2d = find_contours(slice_img, level=label - 0.5)
        for contour in contours_2d:
            contour_3d = np.zeros((contour.shape[0], 3))
            contour_3d[:, 0] = contour[:, 0]  # x
            contour_3d[:, 1] = y              # y
            contour_3d[:, 2] = contour[:, 1]  # z
            contours.append(contour_3d)
    return contours

def visualize_3d_vectors(labels_3d, displacement_mapping, def_properties_npz, output_file, 
                        azim=45, elev=30, length_scale=2, cmap_name='jet', contour_mode='circle',
                        max_particles=100):
    """
    显示 3D 位移向量图，支持两种颗粒轮廓模式
    参数：
        labels_3d: 3D 标签图
        displacement_mapping: {label: disp (3,)} in mm
        def_properties_npz: 颗粒属性文件
        output_file: 输出图像路径
        azim: 水平视角（可调，默认 45）
        elev: 垂直视角（可调，默认 30）
        length_scale: 向量长度缩放（可调，默认 2，适配位移范围 4.633 mm）
        cmap_name: 颜色映射（可调，默认 'jet'，可选 'custom'）
        contour_mode: 轮廓模式（'circle' 或 'boundary'）
        max_particles: 最大颗粒数（仅 boundary 模式，优化性能，默认 100）
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置灰色背景（与 2D 向量模式一致）
    ax.set_facecolor((0.5, 0.5, 0.5))
    fig.patch.set_facecolor((0.5, 0.5, 0.5))
    
    # 加载质心 (centroid(index))
    centroids = load_properties(def_properties_npz)
    
    # 计算最大位移（不归一化）
    vmax = get_max_displacement(displacement_mapping)
    vmin = 0
    norm = plt.Normalize(vmin, vmax)
    
    # 颜色映射（可调）
    cmap = plt.get_cmap(cmap_name)
    
    # 准备数据
    points = []
    vectors = []
    mags = []
    labels = []
    for label, disp in displacement_mapping.items():
        if label in centroids:
            points.append(centroids[label])  # 像素坐标
            vectors.append(disp)             # 物理坐标 (mm)
            mags.append(np.linalg.norm(disp))
            labels.append(label)
    points = np.array(points)
    vectors = np.array(vectors)
    mags = np.array(mags)
    
    # 打印质心范围（像素单位）
    print(f"Centroid(index) ranges - X: [{min(points[:, 0]):.1f}, {max(points[:, 0]):.1f}], "
          f"Y: [{min(points[:, 1]):.1f}, {max(points[:, 1]):.1f}], "
          f"Z: [{min(points[:, 2]):.1f}, {max(points[:, 2]):.1f}]")
    
    # 转换为物理单位（可选，当前保持像素）
    resolution = 0.033321  # mm/pixel
    points_mm = points * resolution  # 转换为 mm（如果需要）
    
    # 绘制颗粒轮廓
    if contour_mode == 'circle':
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   s=10, facecolors='none', edgecolors='lightgray', linewidth=0.5)
    elif contour_mode == 'boundary':
        for i, label in enumerate(labels[:max_particles]):
            contours = get_particle_contours(labels_3d, label)
            for contour in contours:
                ax.plot(contour[:, 0], contour[:, 1], contour[:, 2], 
                        color='lightgray', linewidth=0.5)
    
    # 绘制向量（物理单位 mm）
    for point, vec, mag in zip(points, vectors, mags):
        ax.quiver(point[0], point[1], point[2], 
                  vec[0]*length_scale, vec[1]*length_scale, vec[2]*length_scale,
                  color=cmap(norm(mag)), arrow_length_ratio=0.1, linewidth=1, 
                  length=1, normalize=False, alpha=0.7)
    
    # 设置轴范围（基于 centroid(index)）
    buffer = 100
    ax.set_xlim(min(points[:, 0]) - buffer, max(points[:, 0]) + buffer)
    ax.set_ylim(min(points[:, 1]) - buffer, max(points[:, 1]) + buffer)
    ax.set_zlim(min(points[:, 2]) - buffer, max(points[:, 2]) + buffer)
    
    # 设置轴比例（禁用等比例，保留数据自然比例）
    ax.set_box_aspect(None)  # 禁用立方体，遵循数据比例
    
    # 设置轴标签（可选转换为 mm）
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Z (pixels)')
    # 如果需要物理单位，取消注释以下行并调整：
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_zlabel('Z (mm)')
    # ax.set_xlim(min(points_mm[:, 0]) - buffer*resolution, max(points_mm[:, 0]) + buffer*resolution)
    # ax.set_ylim(min(points_mm[:, 1]) - buffer*resolution, max(points_mm[:, 1]) + buffer*resolution)
    # ax.set_zlim(min(points_mm[:, 2]) - buffer*resolution, max(points_mm[:, 2]) + buffer*resolution)
    
    # 设置视角
    ax.view_init(azim=azim, elev=elev)
    
    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label='Displacement Magnitude (mm)')
    
    # 标题
    def_type = os.path.basename(output_file).split('_')[0]
    plt.title(f'3D Displacement Vectors (Load{def_type[-2:]})')
    
    # 保存图像
    plt.savefig(output_file, dpi=300, facecolor=(0.5, 0.5, 0.5), edgecolor='none')
    plt.show()

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Visualize 3D displacement vectors from 3D labels.")
    parser.add_argument('--def_type', required=True, help='变形体系类型，例如 load1, load5, load10, load15')
    parser.add_argument('--tif_file', type=str, default=None, help='Path to the 3D label TIFF file.')
    parser.add_argument('--results_npz_file', type=str, default=None, help='Path to the results NPZ file.')
    parser.add_argument('--def_properties_npz', type=str, default=None, help='Path to def properties npz file.')
    parser.add_argument('--ref_properties_npz', type=str, default=None, help='Path to ref properties npz file.')
    parser.add_argument('--output_image_dir', type=str, default=None, help='Path to save output image.')
    parser.add_argument('--azim', type=float, default=45, help='Horizontal view angle (degrees)')
    parser.add_argument('--elev', type=float, default=30, help='Vertical view angle (degrees)')
    parser.add_argument('--length_scale', type=float, default=2, help='Vector length scale factor')
    parser.add_argument('--cmap_name', type=str, default='jet', help='Colormap name (e.g., jet, custom)')
    parser.add_argument('--contour_mode', type=str, default='circle', choices=['circle', 'boundary'],
                        help='Contour mode: circle (centroid hollow circle) or boundary (real particle boundary)')
    parser.add_argument('--max_particles', type=int, default=100, help='Max particles for boundary mode')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    type_list = ['origin', 'load1', 'load5', 'load10', 'load15']
    def_index = type_list.index(args.def_type)
    ref_index = def_index - 1
    if ref_index < 0:
        raise ValueError("def_type must be one of 'load1', 'load5', 'load10', 'load15' and cannot be 'origin'.")
    ref_type = type_list[ref_index]
    print(f"Generating 3D vector image: def_type: {args.def_type}, ref_type: {ref_type}, contour_mode: {args.contour_mode}")
    
    if args.tif_file is None:
        args.tif_file = f"/share/home/202321008879/data/sandlabel/{args.def_type}_ai_label.tif"
    if args.results_npz_file is None:
        args.results_npz_file = f"/share/home/202321008879/data/track_results/{args.def_type}to{ref_type}/final_results.npz"
    if args.def_properties_npz is None:
        args.def_properties_npz = f"/share/home/202321008879/data/sand_propertities/{args.def_type}_ai_particle_properties/{args.def_type}_particle_properties.npz"
    if args.ref_properties_npz is None:
        args.ref_properties_npz = f"/share/home/202321008879/data/sand_propertities/{ref_type}_ai_particle_properties/{ref_type}_particle_properties.npz"
    if args.output_image_dir is None:
        args.output_image_dir = f"/share/home/202321008879/data/visualization/3d_vectors/{args.def_type}"
    os.makedirs(args.output_image_dir, exist_ok=True)
    
    labels_3d = tiff.imread(args.tif_file)
    print(f"3D labels shape: {labels_3d.shape}")
    labels_3d = np.transpose(labels_3d, (2, 1, 0)).copy()
    print(f"Transposed labels shape: {labels_3d.shape}")
    
    displacement_mapping = compute_displacement_mapping_from_results_and_properties(
        args.results_npz_file, args.def_properties_npz, args.ref_properties_npz
    )
    
    output_file = os.path.join(args.output_image_dir, f"{args.def_type}_3d_vectors_{args.contour_mode}.png")
    visualize_3d_vectors(
        labels_3d, displacement_mapping, args.def_properties_npz, output_file,
        azim=args.azim, elev=args.elev, length_scale=args.length_scale, 
        cmap_name=args.cmap_name, contour_mode=args.contour_mode, 
        max_particles=args.max_particles
    )
    
    vmax = get_max_displacement(displacement_mapping)
    print(f"Maximum displacement magnitude: {vmax:.3f} mm")