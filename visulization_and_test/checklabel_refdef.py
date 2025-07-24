# 该脚本用于可视化ref和def体系中特定的标签，并进行对比

import sys
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

def read_tif(image_path):
    # 读取tif图像，并将轴排列为 (x, y, z)
    img = tiff.imread(image_path)
    if img.ndim == 3:
        # 假设原tif文件轴顺序为 (z, y, x)
        img = np.transpose(img, (2, 1, 0))
    return img

def extract_label(img, label_value):
    # 根据特定标签生成二值化mask
    mask = np.where(img == label_value, 1, 0)
    return mask

def visualize_comparison(ref_img, def_img, ref_label, def_label):
    # 分别提取不同标签对应的3D二值mask
    ref_mask = extract_label(ref_img, ref_label)
    def_mask = extract_label(def_img, def_label)
    
    from skimage.measure import marching_cubes
    
    # 使用 marching_cubes 提取表面网格
    verts_ref, faces_ref, _, _ = marching_cubes(ref_mask, level=0)
    verts_def, faces_def, _, _ = marching_cubes(def_mask, level=0)
    
    fig = plt.figure(figsize=(12, 6))
    ax_ref = fig.add_subplot(121, projection='3d')
    ax_def = fig.add_subplot(122, projection='3d')
    
    ax_ref.plot_trisurf(verts_ref[:, 0], verts_ref[:, 1], faces_ref, verts_ref[:, 2],
                         color='cyan', lw=1, antialiased=True)
    ax_ref.set_title(f"ref label {ref_label}")
    ax_def.plot_trisurf(verts_def[:, 0], verts_def[:, 1], faces_def, verts_def[:, 2],
                         color='magenta', lw=1, antialiased=True)
    ax_def.set_title(f"def label {def_label}")
    
    plt.tight_layout()
    # 保存图像到文件（例如保存为 PNG 文件）
    plt.savefig("comparison_result.png")
    # 如果在支持GUI的环境下，可使用 plt.show() 显示图像
    # plt.show()

def main():
    # 手动指定参数，修改为实际的文件路径和标签值
    ref_path = "/share/home/202321008879/data/sandlabel/origin_ai_label.tif"  # 修改为实际的ref路径
    def_path = "/share/home/202321008879/data/sandlabel/load1_ai_label.tif"  # 修改为实际的def路径
    ref_label = 14285  # 修改为实际的ref标签值
    def_label = 24762  # 修改为实际的def标签值
        
    ref_img = read_tif(ref_path)
    def_img = read_tif(def_path)
    visualize_comparison(ref_img, def_img, ref_label, def_label)

if __name__ == "__main__":
    main()
