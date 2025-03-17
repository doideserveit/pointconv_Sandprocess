import os
import numpy as np

def check_point_clouds(data_dir, npoints=1200, output_file="insufficient_points.txt"):
    """
    检查生成的 .npz 数据文件，找出点云数量不足 npoints 的文件。
    Args:
        data_dir (str): 数据目录（包含 train 和 test 子目录）。
        npoints (int): 点云数量的最小要求。
        output_file (str): 输出包含点云不足文件的路径列表的文本文件。
    """
    insufficient_files = []  # 存储点云不足的文件路径

    # 遍历 train 和 test 目录
    for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Directory '{split_dir}' does not exist. Skipping...")
            continue

        # 遍历每个类别的目录
        for class_dir in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_dir)
            if not os.path.isdir(class_path):
                continue

            # 遍历 .npz 文件
            for npz_file in os.listdir(class_path):
                if npz_file.endswith(".npz"):
                    npz_path = os.path.join(class_path, npz_file)

                    try:
                        # 加载 .npz 文件
                        data = np.load(npz_path)
                        subsets = data['subsets']

                        # 检查每个子集的点数是否足够
                        for i, subset in enumerate(subsets):
                            if subset.shape[0] < npoints:
                                print(f"File '{npz_path}' - Subset {i} has insufficient points: {subset.shape[0]} < {npoints}")
                                insufficient_files.append(npz_path)
                                break  # 一个文件中有不足点云的子集就记录

                    except Exception as e:
                        print(f"Error loading file '{npz_path}': {e}")
                        insufficient_files.append(npz_path)

    # 将点云不足的文件写入输出文件
    with open(output_file, "w") as f:
        for file in insufficient_files:
            f.write(f"{file}\n")

    print(f"Total insufficient files: {len(insufficient_files)}")
    print(f"Insufficient files saved to '{output_file}'")


if __name__ == "__main__":
    # 数据目录路径
    data_dir = "I:/sandprocess/data/originbad/fps_subsets"
    # 点云数量的最小要求
    npoints = 1200
    # 输出包含不足点云的文件路径的文件
    output_file = "I:/sandprocess/data/originbad/fps_subsets/insufficient_points.txt"

    # 检查点云数据
    check_point_clouds(data_dir, npoints, output_file)
