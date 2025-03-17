import os
import subprocess
import argparse


def count_classes(train_dir):
    """
    Count the number of classes based on the subdirectories in the training dataset folder.
    """
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory '{train_dir}' does not exist!")
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    return len(class_dirs)


def main():
    parser = argparse.ArgumentParser(description="Run data generation and training scripts.")

    # 数据生成参数
    parser.add_argument('--tif_file', type=str, default='I:/sandprocess/data/origintest/origintest.label.tif',
                        help='Path to the input .tif file')
    parser.add_argument('--output_dir', type=str, default='I:/sandprocess/data/origintest/fps_subsets',
                        help='Path to save the generated subsets')
    parser.add_argument('--num_subsets', type=int, default=30, help='Number of subsets to generate per class')
    parser.add_argument('--num_points_per_subset', type=int, default=1200, help='Number of points per subset')
    parser.add_argument('--train_ratio', type=float, default=0.85, help='Train/test split ratio')
    parser.add_argument('--n_jobs', type=int, default=14, help='Number of parallel jobs for processing')

    # 训练脚本参数
    parser.add_argument('--batchsize', type=int, default=10, help='Batch size for training')
    parser.add_argument('--epoch', type=int, default=9, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information')

    args = parser.parse_args()

    # 检查输入路径是否存在
    if not os.path.exists(args.tif_file):
        raise FileNotFoundError(f"The specified tif_file '{args.tif_file}' does not exist!")

    # 调用数据生成脚本
    data_gen_cmd = [
        "python", "tiftrans_direct.py",
        "--tif_file", args.tif_file,
        "--output_dir", args.output_dir,
        "--num_subsets", str(args.num_subsets),
        "--num_points_per_subset", str(args.num_points_per_subset),
        "--train_ratio", str(args.train_ratio),
        "--n_jobs", str(args.n_jobs)
    ]
    try:
        print("Running data generation script...")
        subprocess.run(data_gen_cmd, check=True)
        print("Data generation completed!")
    except subprocess.CalledProcessError as e:
        print(f"Error during data generation: {e}")
        return

    # 确定训练和测试文件夹路径
    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")

    # 动态确定分类数
    try:
        num_classes = count_classes(train_dir)
        print(f"Detected {num_classes} classes in training dataset.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 调用训练脚本
    train_cmd = [
        "python", "train_cls_conv.py",
        "--train_path", train_dir,
        "--test_path", test_dir,
        "--num_classes", str(num_classes),
        "--batchsize", str(args.batchsize),
        "--epoch", str(args.epoch),
        "--learning_rate", str(args.learning_rate),
        "--num_point", str(args.num_points_per_subset),
    ]
    if args.normal:
        train_cmd.append("--normal")  # 添加 normal 参数

    try:
        print("Running training script...")
        subprocess.run(train_cmd, check=True)
        print("Training completed!")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    main()
