import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
import csv
from matplotlib.colors import LinearSegmentedColormap
import scipy.ndimage

def load_rptg_csv(csv_file):
    """读取RPTG csv，返回label到RPTG值的映射"""
    mapping = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = int(row['def_lab'])
            val = row['RPTG']
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



def visualize_rptg_slice(slice_labels, rptg_mapping, output_file, vmin, vmax, axis_label):
    # 以灰度为底图，颗粒区域根据RPTG上色，label=0区域为白色
    plt.figure(figsize=(10, 8))
    # 灰度底图（label>0为灰色，label=0为白色）
    base_img = np.ones_like(slice_labels, dtype=np.float32)  # 全白
    base_img[slice_labels == 0] = 0  # 背景黑色，注释掉即可变为白色
    base_img[slice_labels > 0] = 0.5  # 颗粒区域灰色
    masked_base = np.transpose(base_img, (1, 0))
    plt.imshow(masked_base, cmap='gray', origin='upper')
    # RPTG上色
    color_img = np.zeros_like(slice_labels, dtype=np.float32)
    mask = np.zeros_like(slice_labels, dtype=bool)
    for label in np.unique(slice_labels):
        if label == 0:
            continue
        if label in rptg_mapping and not np.isnan(rptg_mapping[label]):
            color_img[slice_labels == label] = rptg_mapping[label]
            mask[slice_labels == label] = True
    colored = np.ma.masked_where(~mask, color_img)
    colored = np.transpose(colored, (1, 0))
    colors = [(0, 0, 0), (1, 0, 0)]  # 黑色到红色
    custom_cmap = LinearSegmentedColormap.from_list('custom_red', colors)
    im = plt.imshow(colored, cmap='jet', vmin=vmin, vmax=vmax, origin='upper', alpha=0.9)  # 修改选用的cmap颜色映射,hot,jet,custom_red
    cbar = plt.colorbar(im, label='RPTG')
    plt.title('RPTG Visualization')
    plt.xlabel(f'{axis_label[0]} (pixels)')
    plt.ylabel(f'{axis_label[1]} (pixels)')
    plt.savefig(output_file, dpi=300)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="可视化每个颗粒的RPTG数据")
    parser.add_argument('--def_type', required=True, help='变形体系类型，例如 origin, load1, load5, load10, load15')
    parser.add_argument('--slice_axis', type=str, default='XZ', choices=['XY', 'YZ', 'XZ'], help='切片方向')
    parser.add_argument('--num_slices', type=int, default=10, help='要显示的切片数量，均匀采样')
    parser.add_argument('--slice_idx', type=int, default=None, help='切片索引（单张时用）')
    parser.add_argument('--data_dir', default='/share/home/202321008879/data/', help='数据根目录')
    parser.add_argument('--output_dir', default=None, help='输出图片目录')
    parser.add_argument('--rotate', action='store_true', help='是否使用旋转后的tif输出图片')
    return parser.parse_args()

def main():
    args = parse_args()
    def_type = args.def_type
    axis = args.slice_axis.upper()
    axis_map = {'XY': 2, 'YZ': 0, 'XZ': 1}
    axis_label_map = {'XY': ('X', 'Y'), 'YZ': ('Y', 'Z'), 'XZ': ('X', 'Z')}
    axis_dim = axis_map[axis]
    axis_label = axis_label_map[axis]

    # 自动补全路径
    if args.rotate:
        tif_file = os.path.join(args.data_dir, 'sandlabel', f'{def_type}_ai_label_transformed.tif')
    else:
        tif_file = os.path.join(args.data_dir, 'sandlabel', f'{def_type}_ai_label.tif')
    ref_list = ['origin', 'load1', 'load5', 'load10', 'load15']
    ref_idx = ref_list.index(def_type) - 1
    ref_type = ref_list[ref_idx]
    rptg_csv = os.path.join(args.data_dir, 'rptg', f'{def_type}to{ref_type}_RPTG.csv')
    if args.output_dir is None:
        sub_dir = 'rptg_rotate' if args.rotate else 'rptg'
        if args.num_slices == 1 and args.slice_idx is None:
            output_dir = os.path.join(args.data_dir, 'visualization', sub_dir, axis)
        else:
            output_dir = os.path.join(args.data_dir, 'visualization', 'test', sub_dir, def_type, axis)
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    labels_3d = tiff.imread(tif_file)
    labels_3d = np.transpose(labels_3d, (2, 1, 0)).copy()
    print(f"Loaded label data shape: {labels_3d.shape}")
    rptg_mapping = load_rptg_csv(rptg_csv)
    org_sample_height = 58.57788148034935
    global_disp = 0.15 * org_sample_height
    for lab, val in rptg_mapping.items():
        rptg_mapping[lab] = val / global_disp  # 归一化RPTG值
    
    shape = labels_3d.shape
    if args.num_slices == 1:
        slice_indices = [args.slice_idx if args.slice_idx is not None else shape[axis_dim] // 2]
    else:
        slice_indices = np.linspace(0, shape[axis_dim] - 1, args.num_slices, dtype=int)
    
    # 统计RPTG范围
    rptg_vals = [v for v in rptg_mapping.values() if not np.isnan(v)]
    vmin = 0
    vmax = 0.02
    for idx in slice_indices:
        slice_img = get_slice_by_axis(labels_3d, axis, idx)
        output_file = os.path.join(output_dir, f"{args.def_type}rptg_{axis}_slice{idx:04d}.png")
        visualize_rptg_slice(slice_img, rptg_mapping, output_file, vmin, vmax, axis_label)
        print(f"Saved: {output_file}")

if __name__ == '__main__':
    main()
