# 这个脚本用于可视化复杂网络参数（如度、聚类系数等）在不同变形体系下的分布情况
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import argparse
import os
import csv
from matplotlib.colors import LinearSegmentedColormap

def load_param_csv(csv_file, param_name):
    """读取网络参数csv，返回label到参数值的映射"""
    mapping = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = int(row['node'])
            val = row[param_name]
            mapping[lab] = float(val) if val != 'nan' else np.nan
    return mapping

def get_slice_by_axis(labels_3d, axis, idx):
    if axis == 'XY':
        return labels_3d[:, :, idx]
    elif axis == 'YZ':
        return labels_3d[idx, :, :]
    elif axis == 'XZ':
        return labels_3d[:, idx, :]
    else:
        raise ValueError("slice_axis must be 'XY', 'YZ', or 'XZ'")

def load_label_mapping(npz_file):
    """从track_results的npz文件加载def标签到ref标签的映射"""
    data = np.load(npz_file, allow_pickle=True)
    results = data['results'].item()
    mapping = {}
    for d_lab, info in results.items():
        mapping[int(d_lab)] = int(info['predicted_reference_label'])
    return mapping

def compute_param_diff(def_param, ref_param, label_map):
    """根据def标签到ref标签的映射，计算参数差值（def-ref）"""
    diff = {}
    for def_lab, ref_lab in label_map.items():
        def_val = def_param.get(def_lab, np.nan)
        ref_val = ref_param.get(ref_lab, np.nan)
        if not np.isnan(def_val) and not np.isnan(ref_val):
            diff[def_lab] = def_val - ref_val
        else:
            diff[def_lab] = np.nan
    return diff

def visualize_param_slice(slice_labels, param_mapping, output_file, vmin, vmax, axis_label, cmap_name='jet', param_name='param'):
    # 灰度底图（label>0为灰色，label=0为白色）
    plt.figure(figsize=(10, 8))
    base_img = np.ones_like(slice_labels, dtype=np.float32)
    base_img[slice_labels == 0] = 0
    base_img[slice_labels > 0] = 0.5
    masked_base = np.transpose(base_img, (1, 0))
    plt.imshow(masked_base, cmap='gray', origin='upper')
    # 参数上色
    color_img = np.zeros_like(slice_labels, dtype=np.float32)
    mask = np.zeros_like(slice_labels, dtype=bool)
    for label in np.unique(slice_labels):
        if label == 0:
            continue
        if label in param_mapping and not np.isnan(param_mapping[label]):
            color_img[slice_labels == label] = param_mapping[label]
            mask[slice_labels == label] = True
    colored = np.ma.masked_where(~mask, color_img)
    colored = np.transpose(colored, (1, 0))
    im = plt.imshow(colored, cmap=cmap_name, vmin=vmin, vmax=vmax, origin='upper', alpha=0.9)
    cbar = plt.colorbar(im, label=param_name)
    plt.title(f'{param_name} Visualization')
    plt.xlabel(f'{axis_label[0]} (pixels)')
    plt.ylabel(f'{axis_label[1]} (pixels)')
    plt.savefig(output_file, dpi=300)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="可视化复杂网络参数（如度、聚类系数等）")
    parser.add_argument('--def_type', required=True, help='变形体系类型，例如 origin, load1, load5, load10, load15')
    parser.add_argument('--param_name', required=True, help='参数名，如 degree, clustering')
    parser.add_argument('--slice_axis', type=str, default='XZ', choices=['XY', 'YZ', 'XZ'], help='切片方向')
    parser.add_argument('--num_slices', type=int, default=10, help='要显示的切片数量，均匀采样')
    parser.add_argument('--slice_idx', type=int, default=None, help='切片索引（单张时用）')
    parser.add_argument('--data_dir', default='/share/home/202321008879/data/', help='数据根目录')
    parser.add_argument('--output_dir', default=None, help='输出图片目录')
    parser.add_argument('--cmap', default='jet', help='参数上色的colormap')
    return parser.parse_args()

def main():
    args = parse_args()
    def_type = args.def_type
    param_name = args.param_name
    axis = args.slice_axis.upper()
    axis_map = {'XY': 2, 'YZ': 0, 'XZ': 1}
    axis_label_map = {'XY': ('X', 'Y'), 'YZ': ('Y', 'Z'), 'XZ': ('X', 'Z')}
    axis_dim = axis_map[axis]
    axis_label = axis_label_map[axis]

    tif_file = os.path.join(args.data_dir, 'sandlabel', f'{def_type}_ai_label_transformed.tif')

    # 判断是否为差值参数
    if param_name.endswith('_diff'):
        base_param = param_name.replace('_diff', '')
        # 自动推断ref_type
        ref_list = ['origin', 'load1', 'load5', 'load10', 'load15']
        ref_idx = ref_list.index(def_type) - 1
        ref_type = ref_list[ref_idx]

        # 加载映射
        npz_file = os.path.join(args.data_dir, 'track_results', f'{def_type}to{ref_type}', 'final_results.npz')
        label_map = load_label_mapping(npz_file)
        # 加载参数
        def_csv = os.path.join(args.data_dir, 'contact_network', def_type, f'{base_param}.csv')
        ref_csv = os.path.join(args.data_dir, 'contact_network', ref_type, f'{base_param}.csv')
        def_param = load_param_csv(def_csv, base_param)
        ref_param = load_param_csv(ref_csv, base_param)
        param_mapping = compute_param_diff(def_param, ref_param, label_map)
        # 输出目录
        out_param_name = param_name
        if args.output_dir is None:
            output_dir = os.path.join(args.data_dir, 'contact_network', out_param_name, f'{def_type}to{ref_type}', axis)
        else:
            output_dir = args.output_dir
    else:
        param_csv = os.path.join(args.data_dir, 'contact_network', def_type, f'{param_name}.csv')
        param_mapping = load_param_csv(param_csv, param_name)
        out_param_name = param_name
        if args.output_dir is None:
            output_dir = os.path.join(args.data_dir, 'contact_network', param_name, def_type, axis)
        else:
            output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    labels_3d = tiff.imread(tif_file)
    labels_3d = np.transpose(labels_3d, (2, 1, 0)).copy()
    print(f"Loaded label data shape: {labels_3d.shape}")
    # 统计参数范围
    param_vals = [v for v in param_mapping.values() if not np.isnan(v)]
    # vmin = min(param_vals) if param_vals else 0
    # vmax = max(param_vals) if param_vals else 1
    vmin =  -0.01
    vmax = 0.01

    shape = labels_3d.shape
    if args.num_slices == 1:
        slice_indices = [args.slice_idx if args.slice_idx is not None else shape[axis_dim] // 2]
    else:
        slice_indices = np.linspace(0, shape[axis_dim] - 1, args.num_slices, dtype=int)

    for idx in slice_indices:
        slice_img = get_slice_by_axis(labels_3d, axis, idx)
        output_file = os.path.join(output_dir, f"{def_type}_{out_param_name}_{axis}_slice{idx:04d}.png")
        visualize_param_slice(slice_img, param_mapping, output_file, vmin, vmax, axis_label, cmap_name=args.cmap, param_name=out_param_name)
        print(f"Saved: {output_file}")

if __name__ == '__main__':
    main()
