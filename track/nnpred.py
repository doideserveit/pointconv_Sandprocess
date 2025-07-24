# 修改后ModelNetdataLoader.py在推理模式下，此时label_map会将load1中多余的颗粒标签的映射设为原始标签
# 从而在输入进行推理时，多出来的标签不会报错；
# 同时这里的脚本是通过数据加载器loader加载后的数据中获取标签的，此时的label_map在能正确地将其映射为原始标签
# 现在只保存HDF5文件，不再保存CSV文件

import sys, os
# 添加项目根目录
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
# 再将 pointconv_pytorch 目录添加到 sys.path，使得其内部的 utils 可作为顶级模块被导入
pointconv_path = os.path.join(project_root, "pointconv_pytorch")
sys.path.insert(0, pointconv_path)

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import csv
import h5py
from pointconv_pytorch.model.pointconv import PointConvDensityClsSsg as PointConvClsSsg
from pointconv_pytorch.data_utils.ModelNetDataLoader import ModelNetDataLoader


def parse_args():
    parser = argparse.ArgumentParser("Track particles using trained model")
    parser.add_argument('--checkpoint', type=str, default='/share/home/202321008879/experiment/load10ai_classes26311_points1200_2025-05-27_12-06/checkpoints/load10ai-0.999399-0098.pth', 
                        help='Reference system model checkpoint')
    # parser.add_argument('--checkpoint', type=str, default='/share/home/202321008879/experiment/originnew_labbotm1k_'
    #                 'classes1000_points1200_2025-03-19_12-38/checkpoints/originnew_labbotm1k-0.995400-0099.pth', 
    #                     help='Reference system model checkpoint')
    # parser.add_argument('--data_root', type=str, default='/share/home/202321008879/data/h5data/load1_selpar/nomapping', help='Root directory of the h5 data')
    parser.add_argument('--data_root', type=str, default='/share/home/202321008879/data/h5data/load15_ai', help='Root directory of the h5 data')
    parser.add_argument('--num_point', type=int, default=1200, help='Number of points to sample from each subset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size during inference')
    parser.add_argument('--sub_per_particle', type=int, default=10, help='Number of subsets per particle')
    # parser.add_argument('--output_file', type=str, default='/share/home/202321008879/data/track_results/load1_selpar>originnew_labbotm1k/predictions.csv', 
    #                     help='CSV filename to save results')
    parser.add_argument('--output_file', type=str, default='/share/home/202321008879/data/track_results/load15toload10/load15_ai_pred.h5', 
                        help='HDF5 filename to save results')  
    parser.add_argument('--gpu', type=int, default=0, help='指定使用的GPU编号')                  
    return parser.parse_args()

def main(args):

    # 加载 checkpoint 中保存的 num_classes 与 label_map
    ckpt = torch.load(args.checkpoint, map_location=f'cuda:{args.gpu}')
    num_class = ckpt.get('num_classes', None)
    label_map = ckpt.get('label_map', None)
    if num_class is None or label_map is None:
        raise ValueError("Checkpoint缺少 num_classes 或 label_map")
    # 新增：构造映射：将预测索引映射为参考体系中的标签
    inverse_label_map = {v: k for k, v in label_map.items()}
    
    # 修改：调用 ModelNetDataLoader加载数据（该数据加载器内部已有归一化等处理）
    dataset = ModelNetDataLoader(root=args.data_root, npoint=args.num_point, split='eval', normal_channel=True, 
                                label_map=label_map, num_classes=num_class, infer=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    # 打印loader中particle_ids的形状
    
    # MODEL LOADING
    classifier = PointConvClsSsg(num_class, label_map)
    classifier.load_state_dict(ckpt['model_state_dict'])
    device = torch.device(f'cuda:{args.gpu}'  if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    classifier.eval()

    total_particles = 0
    particle_cache = {}

    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    # 只打开 HDF5 文件流
    with h5py.File(args.output_file, 'w') as h5f:
        grp = h5f.create_group("particles")

        # 逐批处理数据并写入文件
        with torch.no_grad():
            for batch_idx, (points, particle_ids) in enumerate(loader):
                # points: (B, npoints, 6) 或 (B, npoints, 3) 根据 normal_channel
                points = points.float().to(device)
                if points.shape[2] == 3:
                    # 未提供法向量，直接拼接占位0
                    zeros = torch.zeros(points.shape[0], points.shape[1], 3, device=device)
                    points = torch.cat([points, zeros], dim=2)
                points = points.permute(0, 2, 1)  # (B,6,npoints)
                xyz = points[:, :3, :]
                normals = points[:, 3:, :]
                outputs = classifier(xyz, normals)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                probs = probs.cpu().numpy()
                preds = preds.cpu().numpy()

                # particle_ids: (B,1) ，先 squeeze 后反映射回原始标签
                particle_ids = particle_ids.squeeze(1)
                orig_particle_ids = np.array([inverse_label_map.get(int(p), int(p)) for p in particle_ids])
                for i in range(len(preds)):
                    particle_id = int(orig_particle_ids[i])
                    mapped_label = inverse_label_map[int(preds[i])]
                    subset_result = {
                        "predicted_label": mapped_label,
                        "probability": probs[i].tolist()
                    }

                    # 计算最终预测标签和平均概率
                    final_index = int(np.argmax(probs[i]))
                    final_label = inverse_label_map[final_index]
                    final_avg_probability = float(probs[i][final_index])

                    # 缓存每个颗粒的所有子集
                    if particle_id not in particle_cache:
                        particle_cache[particle_id] = {
                            "subset_predicted_label": [],
                            "subset_probability": [],
                        }
                    particle_cache[particle_id]["subset_predicted_label"].append(mapped_label)
                    particle_cache[particle_id]["subset_probability"].append(subset_result["probability"])

                    # 如果该颗粒已收集到指定子集数，立即写入HDF5
                    if len(particle_cache[particle_id]["subset_predicted_label"]) == args.sub_per_particle:
                        # 写入HDF5
                        g = grp.create_group(str(particle_id))
                        g.create_dataset("subset_predicted_label", data=np.array(particle_cache[particle_id]["subset_predicted_label"], dtype=np.int32))
                        g.create_dataset("subset_probability", data=np.array(particle_cache[particle_id]["subset_probability"], dtype=np.float32))
                        g.attrs["final_predicted_label"] = final_label
                        g.attrs["final_avg_probability"] = final_avg_probability

                        del particle_cache[particle_id]
                        total_particles += 1
                        if total_particles % 1000 == 0:
                            print(f"Processed {total_particles} particles...")

    print(f"Tracking results saved to HDF5 file: {args.output_file}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
