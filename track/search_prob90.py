# 寻找某阶段PointConv输出概率不低于0.9但未被选为锚点的颗粒
import os
import argparse
import csv
import numpy as np
import h5py

def get_prediction_for_particle_from_h5(h5_path, pending_particle):
    """输入待匹配颗粒标签，返回其预测结果字典"""
    with h5py.File(h5_path, 'r') as h5f:
        particles_grp = h5f['particles']
        pid = str(pending_particle)
        if pid not in particles_grp:
            print(f"Warning: Particle {pending_particle} not found in predictions.")
            return None
        g = particles_grp[pid]
        subsets = []
        for j in range(len(g["subset_predicted_label"])):
            subsets.append({
                "predicted_label": int(g["subset_predicted_label"][j]),
                "probability": g["subset_probability"][j].tolist()
            })
        return {
            "subsets": subsets,
            "final_predicted_label": int(g.attrs["final_predicted_label"]),
            "final_avg_probability": float(g.attrs["final_avg_probability"])
        }
    
def read_initmatch(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data['pending_particles'].tolist()

def main():
    parser = argparse.ArgumentParser(description="寻找某阶段PointConv输出概率不低于0.9但未被选为锚点的颗粒")
    parser.add_argument('--def_type', required=True, help='变形体系类型，例如 origin, load1, load5, load10, load15')
    parser.add_argument('--ref_type', required=True, help='参考体系类型')
    parser.add_argument('--initmatch_npz', default=None, help='initmatch.npz文件路径')
    parser.add_argument('--pred_h5', default=None, help='预测结果 h5 文件路径')
    parser.add_argument('--output_csv', default=None, help='输出csv路径')
    args = parser.parse_args()

    data_dir = '/share/home/202321008879/data/'
    if args.initmatch_npz is None:
        args.initmatch_npz = os.path.join(data_dir, 'track_results', f'{args.def_type}to{args.ref_type}', f'{args.def_type}to{args.ref_type}_initmatch.npz')
    if args.pred_h5 is None:
        args.pred_h5 = os.path.join(data_dir, 'track_results', f'{args.def_type}to{args.ref_type}', f'{args.def_type}_ai_pred.h5')
    if args.output_csv is None:
        args.output_csv = os.path.join(data_dir, 'track_results', f'{args.def_type}to{args.ref_type}', f'prob90_unmatched_{args.def_type}.csv')

    pending_particles = read_initmatch(args.initmatch_npz)
    results = []
    for def_lab in pending_particles:
        pred_info = get_prediction_for_particle_from_h5(args.pred_h5, def_lab)
        if pred_info is None:
            continue
        # 对每个待匹配颗粒，检查其最终概率，若>=0.9，则记录
        if pred_info['final_avg_probability'] >= 0.9:
            results.append((def_lab, pred_info['final_predicted_label'], pred_info['final_avg_probability']))    
    print(f"Found {len(results)} particles with probability >= 0.9")
    # 保存结果到CSV
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['def_label', 'predicted_ref_label', 'final_avg_probability'])
        for row in results:
            writer.writerow(row)
    print(f"Results saved to {args.output_csv}")


if __name__ == '__main__':
    main()