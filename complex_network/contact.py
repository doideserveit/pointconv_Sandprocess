# 该脚本检测接触颗粒
import numpy as np
from tifffile import imread
from joblib import Parallel, delayed
import os
import argparse

def _find_contacts_chunk(coords_chunk, label_volume, offsets, shape, chunk_idx=None, total_chunks=None, total_voxels=None, processed_base=0):
    contacts = set()
    chunk_len = len(coords_chunk)
    for idx, (x, y, z) in enumerate(coords_chunk):
        label_a = label_volume[x, y, z]
        for dx, dy, dz in offsets:
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                label_b = label_volume[nx, ny, nz]
                if label_b == 0 or label_b == label_a:
                    continue
                pair = tuple(sorted((label_a, label_b)))
                contacts.add(pair)
        # 在每个分块内部打印进度
        if (idx + 1) % 100000 == 0 or (idx + 1) == chunk_len:
            global_processed = processed_base + idx + 1
            print(f"Chunk {chunk_idx+1}/{total_chunks}: {idx+1}/{chunk_len} 完成，"
                  f"全局进度: {global_processed}/{total_voxels} ({global_processed/total_voxels*100:.2f}%)")
    return contacts

def find_contacts(label_volume, neighbor_type='26', n_jobs=-1):
    if neighbor_type == '6':
        offsets = [
            (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
        ]
    else:
        offsets = [
            (dx,dy,dz)
            for dx in [-1,0,1]
            for dy in [-1,0,1]
            for dz in [-1,0,1]
            if not (dx==0 and dy==0 and dz==0)
        ]
    shape = label_volume.shape
    coords = np.argwhere(label_volume != 0)
    total_voxels = len(coords)
    print(f"开始检测接触，非零体素数: {total_voxels}")

    # 分块
    chunk_size = 1000000
    chunks = [coords[i:i+chunk_size] for i in range(0, total_voxels, chunk_size)]
    total_chunks = len(chunks)
    processed_bases = [i*chunk_size for i in range(total_chunks)]

    # 并行处理，每个chunk处理完后主进程合并
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_find_contacts_chunk)(
            chunk, label_volume, offsets, shape, idx, total_chunks, total_voxels, processed_bases[idx]
        )
        for idx, chunk in enumerate(chunks)
    )

    # 合并所有进程的结果
    contacts = set()
    for s in results:
        contacts.update(s)
    print("接触检测完成。")
    return contacts

def parse_args():
    parser = argparse.ArgumentParser(description="检测接触颗粒")
    parser.add_argument('--def_type', type=str, required=True,
                        help='变形体系类型，例如 origin, load1, load5, load10, load15')
    parser.add_argument('--tif_path', type=str, help='输入的TIF文件路径')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--neighbor_type', type=str, choices=['6', '26'], default='26',
                        help='邻居类型，6或26连接')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='并行处理的线程数，-1表示使用所有可用线程')
    return parser.parse_args()

def main():
    args = parse_args()
    # 自动补全文件路径
    if args.tif_path is None:
        args.tif_path = f'/share/home/202321008879/data/sandlabel/{args.def_type}_ai_label.tif'
    if args.output_dir is None:
        args.output_dir = f'/share/home/202321008879/data/contact_network'

   
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{args.def_type}_contacts.txt')

    # 读取TIF文件
    label_volume = imread(args.tif_path)
    print(f"读取TIF文件: {args.tif_path}, 体素形状: {label_volume.shape}")

    # 检测接触颗粒
    contacts = find_contacts(label_volume, neighbor_type=args.neighbor_type, n_jobs=args.n_jobs)
    print(f"共找到{len(contacts)}对接触颗粒")

    # 输出到文件
    with open(output_file, 'w') as f:
        for a, b in sorted(contacts):
            f.write(f"{a},{b}\n")
    print(f"接触信息已保存到: {output_file}")

if __name__ == "__main__":
    main()


# if __name__ == '__main__':
#     # 示例用法
#     tif_path = '/share/home/202321008879/data/sandlabel/origin_ai_label.tif'  # 替换为你的tif路径
#     output_dir = '/share/home/202321008879/data/contact_network'  # 输出文件路径
#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, 'contacts.txt')

#     label_volume = imread(tif_path)
#     # 若需要转置轴顺序，按实际情况调整
#     # label_volume = np.transpose(label_volume, (2,1,0)).copy()
#     contacts = find_contacts(label_volume, neighbor_type='26')
#     print(f"共找到{len(contacts)}对接触颗粒")
#     # 输出到文件
#     with open(output_file, 'w') as f:
#         for a, b in sorted(contacts):
#             f.write(f"{a},{b}\n")