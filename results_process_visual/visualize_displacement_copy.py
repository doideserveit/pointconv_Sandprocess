# 向量模式：默认选择当前切片的质心作为起点，位移向量的分量根据切片方向自动选择。例如YZ切片，则使用YZ位移分量。
# 上色模式：可以选择使用位移的X、Y、Z分量或模长进行上色。（指定color_mode）
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import math
import os
import argparse
import matplotlib as mpl

def load_properties(npz_path):
    """
    加载颗粒属性文件，返回 {label: centroid}
    """
    data = np.load(npz_path, allow_pickle=True)
    labels = data['label']
    centroids = data['corrected_centroid(physical)']
    mapping = {}
    for i, lab in enumerate(labels):
        mapping[int(lab)] = np.array(centroids[i])
    return mapping

def compute_displacement_mapping_from_results_and_properties(results_npz_file, def_properties_npz, ref_properties_npz):
    """
    从results_npz_file中获取def label和其对应的ref label，
    然后分别从def/ref的properties中获取质心，计算位移（def质心 - ref质心）。
    返回: {def_label: displacement (3,)}
    """
    def_mapping = load_properties(def_properties_npz)
    ref_mapping = load_properties(ref_properties_npz)
    data = np.load(results_npz_file, allow_pickle=True)
    results = data['results'].item()
    displacement_mapping = {}
    for def_label, info in results.items():
        ref_label = info.get('predicted_reference_label', None)
        if ref_label is None:
            print(f"Warning: def_label {def_label} has no 'predicted_reference_label' in results, skipped.")
            continue
        def_label = int(def_label)
        ref_label = int(ref_label)
        if def_label not in def_mapping:
            print(f"Warning: def_label {def_label} not found in def properties, skipped.")
            continue
        if ref_label not in ref_mapping:
            print(f"Warning: ref_label {ref_label} not found in ref properties, skipped.")
            continue
        def_centroid = def_mapping[def_label]
        ref_centroid = ref_mapping[ref_label]
        disp = np.array(def_centroid) - np.array(ref_centroid)
        if disp.shape != (3,):
            print(f"Warning: Label {def_label} displacement shape {disp.shape} is not (3,). Skipped.")
            continue
        displacement_mapping[def_label] = disp
    if len(displacement_mapping) == 0:
        print("Error: No valid displacement mapping found. Please check your results file and properties files.")
    return displacement_mapping

def map_labels_to_displacement(slice_labels, displacement_mapping, color_mode='mag', slice_axis='XY'):
    """
    用于上色模式获取位移数据。
    color_mode: 'x', 'y', 'z', 'mag'
    """
    displacement_slice = np.zeros_like(slice_labels, dtype=np.float32)
    labels_not_found = set()
    for label in np.unique(slice_labels):
        if label == 0:
            continue
        if label in displacement_mapping:
            disp = displacement_mapping[label]
            if color_mode == 'x':
                value = disp[0]
            elif color_mode == 'y':
                value = disp[1]
            elif color_mode == 'z':
                value = disp[2]
            elif color_mode == 'mag':
                value = np.linalg.norm(disp)
            else:
                value = np.linalg.norm(disp)
            displacement_slice[slice_labels == label] = value
        else:
            labels_not_found.add(label)
            print(f"Warning: Label {label} not found in displacement mapping.")
    return displacement_slice, labels_not_found

def get_info_in_slice(labels_3d, slice_axis, slice_idx):
    """
    只对当前切片中实际出现的颗粒label计算质心（在切片平面上的投影）。
    返回:
    字典centroids: {label: (x, y) 或 (y, z) 或 (x, z)}，根据切片方向
    slice_img: 对应轴的二维切片图像
    """
    centroids = {}
    if slice_axis == 'XY':

        # XY平面，Z为切片索引，shape 
        slice_img = labels_3d[:, :, slice_idx]  # shape (Y, X)（因为横轴是X，所以每一列的index是x）
        # 转置(plt.show时，第1轴作为图像中的x轴，第0轴作为y轴，其实就是跟数组形状对应来画图)
        slice_img = np.transpose(slice_img, (1, 0)) 
        labels_in_slice = np.unique(slice_img)
        labels_in_slice = labels_in_slice[labels_in_slice != 0]
        for label in labels_in_slice:
            coords = np.argwhere(slice_img == label)  # coords: (N, 2), 是N个像素的坐标（对应label像素的行号，列号） (y, x)
            if coords.shape[0] == 0:
                continue
            centroid_yx = coords.mean(axis=0)
            centroid = (centroid_yx[1], centroid_yx[0])
            centroids[label] = centroid
    elif slice_axis == 'YZ':
        slice_img = labels_3d[slice_idx, :, :]
        slice_img = np.transpose(slice_img, (1, 0))  # 转置
        labels_in_slice = np.unique(slice_img)
        labels_in_slice = labels_in_slice[labels_in_slice != 0]
        for label in labels_in_slice:
            coords = np.argwhere(slice_img == label)  # coords: (N, 3), (z, y)
            if coords.shape[0] == 0:
                continue
            centroid_zy = coords.mean(axis=0)  
            centroid = (centroid_zy[1], centroid_zy[0])  # (y, z)
            centroids[label] = centroid
    elif slice_axis == 'XZ':
        slice_img = labels_3d[:, slice_idx, :]
        slice_img = np.transpose(slice_img, (1, 0))  # 转置
        labels_in_slice = np.unique(slice_img)
        labels_in_slice = labels_in_slice[labels_in_slice != 0]
        for label in labels_in_slice:
            coords = np.argwhere(slice_img == label)  # coords: (N, 2), (z, x)
            if coords.shape[0] == 0:
                continue
            centroid_zx = coords.mean(axis=0)  # (z, x)
            centroid = (centroid_zx[1], centroid_zx[0])  # (x, z)
            centroids[label] = centroid
    else:
        pass
    return centroids, slice_img

def get_custom_cmap():
    """
    构造一个自定义colormap，超过1.6的部分为红色，颜色过渡自然
    """
    base_cmap = plt.get_cmap('jet')
    colors = base_cmap(np.linspace(0, 1, 256))
    # 将超过1.6的部分设置为红色
    threshold = 1.6 / 1.8  # 归一化到1.8范围内
    colors[int(threshold * 256):] = [1, 0, 0, 1]  # 红色
    custom_cmap = mpl.colors.ListedColormap(colors)
    return custom_cmap

def visualize_displacement(slice_displacement, output_image_file, vmin, vmax, slice_axis='XY', use_custom_cmap=False):
    """
    上色模式显示
    use_custom_cmap: 归一化模式下为True
    """
    plt.figure(figsize=(10, 8))
    if use_custom_cmap:
        display_vmax = 1.8  # colorbar显示到1.8
        norm = mpl.colors.Normalize(vmin=vmin, vmax=display_vmax)
        display_data = np.clip(slice_displacement, vmin, 1.6)  # 大于 1.6 的值设置为 1.6
        masked = np.ma.masked_where(display_data == 0, display_data)
        masked = np.transpose(masked, (1, 0))
        cmap = get_custom_cmap()
        cmap.set_bad(color='white')
        im = plt.imshow(masked, cmap=cmap, norm=norm, origin='upper')
        cbar = plt.colorbar(im)
        cbar.set_label('Displacement (normalized)')
        ticks = [0, 0.5, 1.0, 1.6]  # 只显示到1.6
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(['0', '0.5', '1.0', '1.6'])
        cbar.ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    else:
        masked = np.ma.masked_where(slice_displacement == 0, slice_displacement)
        masked = np.transpose(masked, (1, 0))
        cmap = plt.get_cmap('jet')
        cmap.set_bad(color='white')
        im = plt.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        plt.colorbar(im, label='Displacement (mm)')
    plt.title('Displacement Visualization')
    if slice_axis == 'XY':
        plt.xlabel('X-axis (pixels)')
        plt.ylabel('Y-axis (pixels)')
    elif slice_axis == 'YZ':
        plt.xlabel('Y-axis (pixels)')
        plt.ylabel('Z-axis (pixels)')
    elif slice_axis == 'XZ':
        plt.xlabel('X-axis (pixels)')
        plt.ylabel('Z-axis (pixels)')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
    plt.savefig(output_image_file, dpi=300)
    plt.show()

def visualize_displacement_vectors(labels_3d, displacement_mapping, slice_axis, slice_idx,
                                   output_image_file, vmin=0, vmax=1, vec_idx=(0, 1), axis_label=('X', 'Y'), use_custom_cmap=False):
    """
    向量模式显示
    use_custom_cmap: 归一化模式下为True
    """
    plt.figure(figsize=(10, 8))
    centroids, slice_img = get_info_in_slice(labels_3d, slice_axis, slice_idx)
    plt.imshow(slice_img, cmap='gray', origin='upper', alpha=0.3)
    if use_custom_cmap:
        display_vmax = 1.8
        norm = mpl.colors.Normalize(vmin=vmin, vmax=display_vmax)
        cmap = get_custom_cmap()
    else:
        norm = plt.Normalize(vmin, vmax)
        cmap = plt.get_cmap('jet')
    points = []
    mags = []
    labels_not_found = set()
    for label, centroid in centroids.items():
        if label not in displacement_mapping:
            labels_not_found.add(label)
            continue
        disp = displacement_mapping[label]
        vec = np.array([disp[vec_idx[0]], disp[vec_idx[1]]])
        mag = np.linalg.norm(vec)
        points.append(centroid)
        mags.append(mag)
    points = np.array(points)
    mags = np.array(mags)
    if len(points) > 0:
        if use_custom_cmap:
            sc = plt.scatter(points[:,0], points[:,1], c=np.clip(mags, vmin, 1.6), cmap=cmap, norm=norm, s=0)
        else:
            sc = plt.scatter(points[:,0], points[:,1], c=mags, cmap=cmap, norm=norm, s=0)
    else:
        sc = None
    for i, (label, centroid) in enumerate(centroids.items()):
        if label not in displacement_mapping:
            continue
        disp = displacement_mapping[label]
        vec = np.array([disp[vec_idx[0]], disp[vec_idx[1]]])
        mag = np.linalg.norm(vec)
        color = cmap(norm(min(mag, 1.6))) if use_custom_cmap else cmap(norm(mag))
        x, y = centroid
        plt.arrow(x, y, vec[0]*10, vec[1]*10, color=color, head_width=2, head_length=2, length_includes_head=True)
    if sc is not None:
        if use_custom_cmap:
            cbar = plt.colorbar(sc)
            cbar.set_label(f'Displacement Magnitude ({axis_label[0]}{axis_label[1]})')
            ticks = [0, 0.5, 1.0, 1.6]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(['0', '0.5', '1.0', '1.6'])
            cbar.ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
        else:
            plt.colorbar(sc, label=f'Displacement Magnitude ({axis_label[0]}{axis_label[1]})')
    plt.title(f'Displacement Vectors (slice {slice_idx}, {axis_label[0]}{axis_label[1]} plane)')
    plt.xlabel(f'{axis_label[0]} (pixels)')
    plt.ylabel(f'{axis_label[1]} (pixels)')
    plt.savefig(output_image_file, dpi=300)
    plt.show()
    if labels_not_found:
        print(f"Warning: {len(labels_not_found)} labels in this slice have no displacement data: {sorted(labels_not_found)}")
    return labels_not_found

def get_slice_by_axis(labels_3d, axis, idx):
    """根据切片方向和索引获取切片"""
    if axis == 'XY':
        return labels_3d[:, :, idx]
    elif axis == 'YZ':
        return labels_3d[idx, :, :]
    elif axis == 'XZ':
        return labels_3d[:, idx, :]
    else:
        raise ValueError("slice_axis must be 'XY', 'YZ', or 'XZ'")

def get_vec_idx_and_label(axis):
    """根据切片方向返回对应的分量索引和标签"""
    if axis == 'XY':
        return (0, 1), 'xy'
    elif axis == 'YZ':
        return (1, 2), 'yz'
    elif axis == 'XZ':
        return (0, 2), 'xz'
    else:
        raise ValueError("slice_axis must be 'XY', 'YZ', or 'XZ'")

def get_max_displacement(displacement_mapping, color_mode='mag', slice_axis='XY'):
    """根据color_mode获取最大位移值"""
    if color_mode == 'x':
        return max(abs(v[0]) for v in displacement_mapping.values())
    elif color_mode == 'y':
        return max(abs(v[1]) for v in displacement_mapping.values())
    elif color_mode == 'z':
        return max(abs(v[2]) for v in displacement_mapping.values())
    elif color_mode == 'mag':
        return max(np.linalg.norm(v) for v in displacement_mapping.values())
    elif color_mode == 'vector':
        vec_idx, _ = get_vec_idx_and_label(slice_axis)
        return max(np.linalg.norm([v[vec_idx[0]], v[vec_idx[1]]]) for v in displacement_mapping.values())
    else:
        return max(np.linalg.norm(v) for v in displacement_mapping.values())

def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="Visualize displacement from 3D labels and displacement mapping.")
    parser.add_argument('--def_type', required=True, help='变形体系类型，例如load1, load5, load10, load15')
    parser.add_argument('--color_mode', type=str, default='mag', choices=['x', 'y', 'z', 'mag', 'vector'],
                        help='上色方式: x/y/z分量、总位移模长、或切片平面分量向量箭头(vector)')
    parser.add_argument('--slice_axis', type=str, default='XZ', choices=['XY', 'YZ', 'XZ'],
                        help='切片方向: XY, YZ, XZ')
    parser.add_argument('--num_slices', type=int, default=10, help='要显示的切片数量，均匀采样')
    parser.add_argument('--norml', action='store_true', help='是否对位移数据进行归一化处理')
    parser.add_argument('--slice_idx', type=int, default=None, help='切片索引（单张时用）')
    parser.add_argument('--tif_file', type=str, default=None, help='Path to the 3D label TIFF file.')
    parser.add_argument('--results_npz_file', type=str, default=None, help='Path to the results NPZ file containing displacement mapping.')
    parser.add_argument('--def_properties_npz', type=str, default=None, help='Path to def properties npz file.')
    parser.add_argument('--ref_properties_npz', type=str, default=None, help='Path to ref properties npz file.')
    parser.add_argument('--output_image_dir', type=str, default=None, help='Path to save the output visualization image.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    type_list = ['origin', 'load1', 'load5', 'load10', 'load15']
    def_index = type_list.index(args.def_type)
    ref_index = def_index - 1
    if ref_index < 0:
        raise ValueError("def_type must be one of 'load1', 'load5', 'load10', 'load15' and cannot be 'origin'.")
    ref_type = type_list[ref_index]
    print(f"Generating image: def_type: {args.def_type}, ref_type: {ref_type}")
    # 全局位移计算
    org_sample_height = 58.57788148034935
    strain = [0, 0.01, 0.05, 0.1, 0.15]
    global_disp = (strain[def_index] - strain[ref_index]) * org_sample_height

    # 自动补全文件路径
    if args.tif_file is None:
        args.tif_file = f"/share/home/202321008879/data/sandlabel/{args.def_type}_ai_label.tif"
    if args.results_npz_file is None:
        args.results_npz_file = f"/share/home/202321008879/data/track_results/{args.def_type}to{ref_type}/final_results.npz"
    if args.def_properties_npz is None:
        args.def_properties_npz = f"/share/home/202321008879/data/sand_propertities/{args.def_type}_ai_particle_properties/{args.def_type}_particle_properties.npz"
    if args.ref_properties_npz is None:
        args.ref_properties_npz = f"/share/home/202321008879/data/sand_propertities/{ref_type}_ai_particle_properties/{ref_type}_particle_properties.npz"    
    if args.output_image_dir is None:
        args.output_image_dir = f"/share/home/202321008879/data/visualization/test/{args.def_type}/{args.slice_axis}_{args.color_mode}"
    os.makedirs(args.output_image_dir, exist_ok=True)

    # 加载 3D 标签图和位移数据
    labels_3d = tiff.imread(args.tif_file)
    print(f"3D labels shape: {labels_3d.shape}")
    labels_3d = np.transpose(labels_3d, (2, 1, 0)).copy()
    print(f"Transposed labels shape: {labels_3d.shape}")

    displacement_mapping = compute_displacement_mapping_from_results_and_properties(
        args.results_npz_file, args.def_properties_npz, args.ref_properties_npz
    )
    if args.norml:
        # 对位移数据进行归一化处理
        for label, disp in displacement_mapping.items():
            displacement_mapping[label] = disp / global_disp

    # 选择切片方向
    axis = args.slice_axis.upper()
    axis_map = {'XY': 2, 'YZ': 0, 'XZ': 1}
    axis_dim = axis_map[axis]
    num_slices = args.num_slices
    shape = labels_3d.shape

    # 计算切片索引列表
    if num_slices == 1:
        if args.slice_idx is not None:
            slice_indices = [args.slice_idx]
        else:
            slice_indices = [shape[axis_dim] // 2]
    else:
        slice_indices = np.linspace(0, shape[axis_dim] - 1, num_slices, dtype=int)

    all_labels_not_found = set()

    if args.color_mode == 'vector':
        vec_idx, vec_label = get_vec_idx_and_label(axis)
        if args.norml:
            vmax_vec = 1.6  # 归一化模式下固定vmax为1.6
        else:
            vmax_vec = get_max_displacement(displacement_mapping, color_mode='vector', slice_axis=axis)

        for idx in slice_indices:
            output_image_file = os.path.join(
                args.output_image_dir,
                f"disp_vector_{vec_label}_{axis}_slice{idx:04d}.png"
            )
            labels_not_found = visualize_displacement_vectors(
                labels_3d, displacement_mapping, axis, idx,
                output_image_file, vmin=0, vmax=vmax_vec, vec_idx=vec_idx, axis_label=(axis[0], axis[1]),
                use_custom_cmap=args.norml
            )
            all_labels_not_found.update(labels_not_found)
    else:
        if args.norml:
            vmax = 1.6  # 归一化模式下固定vmax为1.6
        else:
            vmax = get_max_displacement(displacement_mapping, color_mode=args.color_mode, slice_axis=axis)

        for idx in slice_indices:
            slice_img = get_slice_by_axis(labels_3d, axis, idx)
            slice_displacement, labels_not_found = map_labels_to_displacement(
                slice_img, displacement_mapping, color_mode=args.color_mode, slice_axis=axis
            )
            all_labels_not_found.update(labels_not_found)
            output_image_file = os.path.join(
                args.output_image_dir,
                f"disp_{args.color_mode}_{axis}_slice{idx:04d}.png"
            )
            visualize_displacement(
                slice_displacement, output_image_file, vmin=0, vmax=vmax, slice_axis=axis,
                use_custom_cmap=args.norml
            )
    print(f"Normalization mode: {args.norml}")
    if args.color_mode == 'vector':
        print(f"Vector mode: vmax_vec = {vmax_vec}")
    else:
        print(f"Color mode: vmax = {vmax}")
    print(f"\n总共有 {len(all_labels_not_found)} 个颗粒在所有切片中出现但没有位移数据。")
    print("这些颗粒的label为：", sorted(all_labels_not_found))