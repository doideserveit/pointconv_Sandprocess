# 该脚本用于查询def_lab对特定标签的预测概率
import numpy as np
import argparse

def query_prediction(predictions_file, def_lab, ref_lab):
    """
    查询 def_lab 对 ref_lab 的预测概率，输出 10 个子集的概率和平均概率。
    """
    # 加载预测文件
    pred_data = np.load(predictions_file, allow_pickle=True)
    predictions = pred_data['results'].item()

    if def_lab not in predictions:
        print(f"def_lab {def_lab} 不存在于预测结果中。")
        return

    subsets = predictions[def_lab]["subsets"]
    ref_probs = [subset["probability"][ref_lab - 1] for subset in subsets]  # ref_lab 从 1 开始编号
    avg_prob = np.mean(ref_probs)

    print(f"def_lab: {def_lab}, ref_lab: {ref_lab}")
    print("子集概率:")
    for i, prob in enumerate(ref_probs):
        print(f"  子集 {i + 1}: {prob:.6f}")
    print(f"平均概率: {avg_prob:.6f}")


def main():
    parser = argparse.ArgumentParser(description="查询 def_lab 对 ref_lab 的预测概率")
    parser.add_argument('--predictions_file', default='/share/home/202321008879/data/track_results/load1_aitoorigin_ai(old)/load1_ai_pred.npz', help="预测结果 npz 文件路径")
    parser.add_argument('--def_lab', type=int, required=True, help="待查询的 def_lab 标签")
    parser.add_argument('--ref_lab', type=int, required=True, help="待查询的 ref_lab 标签")
    args = parser.parse_args()

    query_prediction(args.predictions_file, args.def_lab, args.ref_lab)

if __name__ == '__main__':
    main()