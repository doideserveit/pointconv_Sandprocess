# 修改后ModelNetdataLoader.py在推理模式下，此时label_map会将load1中多余的颗粒标签的映射设为原始标签
# 从而在输入进行推理时，多出来的标签不会报错；
# 同时这里的脚本是通过数据加载器loader加载后的数据中获取标签的，此时的label_map在能正确地将其映射为原始标签

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
from pointconv_pytorch.model.pointconv import PointConvDensityClsSsg as PointConvClsSsg
from pointconv_pytorch.data_utils.ModelNetDataLoader import ModelNetDataLoader


def parse_args():
    parser = argparse.ArgumentParser("Track particles using trained model")
    parser.add_argument('--checkpoint', type=str, default='/share/home/202321008879/experiment/originai_classes25333_points1200_2025-05-02_11-25/checkpoints/originai-0.999921-0100.pth', 
                        help='Reference system model checkpoint')
    # parser.add_argument('--checkpoint', type=str, default='/share/home/202321008879/experiment/originnew_labbotm1k_'
    #                 'classes1000_points1200_2025-03-19_12-38/checkpoints/originnew_labbotm1k-0.995400-0099.pth', 
    #                     help='Reference system model checkpoint')
    # parser.add_argument('--data_root', type=str, default='/share/home/202321008879/data/h5data/load1_selpar/nomapping', help='Root directory of the h5 data')
    parser.add_argument('--data_root', type=str, default='/share/home/202321008879/data/h5data/load1_ai', help='Root directory of the h5 data')
    parser.add_argument('--num_point', type=int, default=1200, help='Number of points to sample from each subset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size during inference')
    parser.add_argument('--sub_per_particle', type=int, default=10, help='Number of subsets per particle')
    # parser.add_argument('--output_file', type=str, default='/share/home/202321008879/data/track_results/load1_selpar>originnew_labbotm1k/predictions.csv', 
    #                     help='CSV filename to save results')
    parser.add_argument('--output_file', type=str, default='/share/home/202321008879/data/track_results/load1toorigin/load1_ai_pred.csv', 
                        help='CSV filename to save results')                    
    return parser.parse_args()

def main(args):
    # 加载 checkpoint 中保存的 num_classes 与 label_map
    ckpt = torch.load(args.checkpoint, map_location='cuda:0')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    classifier.eval()

    results = {}  # 按颗粒聚合结果，粒子ID直接来源于 h5 数据
    with torch.no_grad():
        for points, particle_ids in loader:
            # points: (B, npoints, 6) 或 (B, npoints, 3) 根据 normal_channel
            points = points.float().cuda()
            if points.shape[2] == 3:
                # 未提供法向量，直接拼接占位0
                zeros = torch.zeros(points.shape[0], points.shape[1], 3).to(points.device)
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
                if particle_id not in results:
                    results[particle_id] = {"subsets": []}
                results[particle_id]["subsets"].append(subset_result)

    for particle_id, info in results.items():
        subset_probs = np.array([s["probability"] for s in info["subsets"]])
        # 先计算每个类别在所有子集中的均值，确定最终预测标签
        avg_prob_all = np.mean(subset_probs, axis=0)
        final_index = int(np.argmax(avg_prob_all))
        final_label = inverse_label_map[final_index]
        # 修改：只计算最终预测标签的平均概率
        final_label_probs = [s["probability"][final_index] for s in info["subsets"]]
        final_avg_probability = float(np.mean(final_label_probs))
        info["final_predicted_label"] = final_label
        info["final_avg_probability"] = final_avg_probability
    # 该 npz 文件只保存了一个顶层键，名称为 "results"。
    # "results" 对应的值是一个字典，其中每个键为粒子的 ID，值为一个字典，包含以下键：
    # "subsets"：值是一个列表，其每个元素为子集的结果字典（其内部包含 "predicted_label" 和 "probability" 键）。
    # "final_predicted_label"：最终预测的标签。
    # "final_avg_probability"：最终预测标签对应的平均概率。

    header = ["Particle_ID"]
    num_subsets = args.sub_per_particle  # 每个颗粒包含sub_per_particle个子集
    for i in range(1, num_subsets+1):
        header.extend([f"subset{i}_predicted_label", f"subset{i}_probability"])
    header.extend(["final_predicted_label", "final_avg_probability"])

    with open(args.output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for particle_id, info in sorted(results.items()):
            row = [particle_id]
            for s in info["subsets"]:
                row.append(s["predicted_label"])
                prob_str = " ".join([f"{p:.4f}" for p in s["probability"]])
                row.append(prob_str)
            row.extend([
                info["final_predicted_label"],
                f"{info['final_avg_probability']:.4f}"
            ])
            writer.writerow(row)
    print(f"Tracking results saved to CSV file: {args.output_file}")

    # 新增：保存结果为 npz 格式
    npz_file = args.output_file.replace('.csv', '.npz')
    np.savez_compressed(npz_file, results=results)
    print(f"Tracking results saved to NPZ file: {npz_file}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
