# Convert npz results to h5 format
import numpy as np
import h5py
import argparse
import os

def main(npz_path, h5_path):
    # 加载npz文件
    data = np.load(npz_path, allow_pickle=True)
    results = data['results'].item()  # 取出字典

    output_dir = os.path.dirname(h5_path)
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, 'w') as h5f:
        grp = h5f.create_group("particles")
        for pid, pdata in results.items():
            g = grp.create_group(str(pid))
            # 收集所有子集的预测标签和概率
            subset_predicted_label = []
            subset_probability = []
            for subset in pdata["subsets"]:
                subset_predicted_label.append(subset["predicted_label"])
                subset_probability.append(subset["probability"])
            g.create_dataset("subset_predicted_label", data=np.array(subset_predicted_label, dtype=np.int32))
            g.create_dataset("subset_probability", data=np.array(subset_probability, dtype=np.float32))
            g.attrs["final_predicted_label"] = pdata["final_predicted_label"]
            g.attrs["final_avg_probability"] = pdata["final_avg_probability"]

    print(f"Converted {npz_path} to {h5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert npz results to h5 format")
    parser.add_argument('--npz', type=str, required=True, help='输入的npz文件路径')
    parser.add_argument('--h5', type=str, required=True, help='输出的h5文件路径')
    args = parser.parse_args()
    main(args.npz, args.h5)
