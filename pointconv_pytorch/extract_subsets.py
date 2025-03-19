# 从原始数据集中提取指定标签范围(1000个)的子集数据，保存到新的 h5 文件中
import h5py
import numpy as np
import os

def load_h5(filename):
    # 载入 h5 文件中的 subsets 和 labels
    with h5py.File(filename, 'r') as f:
        subsets = f['subsets'][:]
        labels = f['labels'][:]
    return subsets, labels

def save_h5(filename, subsets, labels):
    # 保存数据到 h5 文件
    with h5py.File(filename, 'w') as f:
        f.create_dataset('subsets', data=subsets, compression="gzip")
        f.create_dataset('labels', data=labels, compression="gzip")
    print(f"Saved extracted data to {filename}")

def extract_label_range(subsets, labels, low=1, high=1000):
    # 根据标签数值过滤范围 [low, high]
    mask = (labels >= low) & (labels <= high)
    extracted_subsets = subsets[mask]
    extracted_labels = labels[mask]
    return extracted_subsets, extracted_labels

def validate_extraction(extracted_subsets, extracted_labels, dataset_name, low, high):
    if dataset_name == "训练数据":
        num_subsets_per_class = 25
        expected_count = (high - low + 1) * num_subsets_per_class
    elif dataset_name == "测试数据":
        num_subsets_per_class = 5
        expected_count = (high - low + 1) * num_subsets_per_class
    elif dataset_name == "评估数据":
        num_subsets_per_class = 10
        expected_count = (high - low + 1) * num_subsets_per_class
    
    actual_count = extracted_subsets.shape[0]
    print(f"\n验证'{dataset_name}':")
    if actual_count == expected_count:
        print(f"总共的子集数为{actual_count}，{dataset_name}所提取的每个标签确实包含 {num_subsets_per_class} 个子集。")
    else:
        print(f"总共的子集数为{actual_count}，{dataset_name}所提取的标签不包含 {num_subsets_per_class} 个子集，请检查数据: {actual_count}")
    
    unique_labels = np.unique(extracted_labels)
    print("标签数量:", extracted_labels.shape[0])
    expected_labels = np.arange(low, high + 1)
    if np.array_equal(unique_labels, expected_labels):
        print(f"所有标签都存在，没有缺少。")
    else:
        missing = np.setdiff1d(expected_labels, unique_labels)
        print("缺少的标签:", missing)
    if unique_labels.size == (high - low + 1):
        print(f"提取的{dataset_name}数据包含 {(high - low + 1)} 个颗粒。")
    else:
        print(f"提取的{dataset_name}数据不包含 {(high - low + 1)} 个独特的颗粒，请检查数据: {unique_labels}")

def main(low, high, base_dir, output_dir):
    # train_file = os.path.join(base_dir, 'all_train_fps_subsets.h5')    # 先注释掉
    # test_file = os.path.join(base_dir, 'all_test_fps_subsets.h5')    # 先注释掉
    eval_file = os.path.join(base_dir, 'all_eval_fps_subsets.h5')

    # # 载入原始训练数据
    # train_subsets, train_labels = load_h5(train_file)
    # extract_train_subsets, extract_train_labels = extract_label_range(train_subsets, train_labels, low, high)  # 先注释掉
    
    # # 载入原始测试数据
    # test_subsets, test_labels = load_h5(test_file)
    # extract_test_subsets, extract_test_labels = extract_label_range(test_subsets, test_labels, low, high)  # 先注释掉
    
    # 加载评估eval数据
    eval_subsets, eval_labels = load_h5(eval_file)
    extract_eval_subsets, extract_eval_labels = extract_label_range(eval_subsets, eval_labels, low, high)
    
    # 定义提取后数据的保存路径
    # out_train_file = os.path.join(output_dir, 'all_train_fps_subsets.h5')   # 先注释掉
    # out_test_file = os.path.join(output_dir, 'all_test_fps_subsets.h5')    # 先注释掉
    out_eval_file = os.path.join(output_dir, 'all_eval_fps_subsets.h5')
    
    # 保存提取后的数据
    # save_h5(out_train_file, extract_train_subsets, extract_train_labels)  # 先注释掉
    # save_h5(out_test_file, extract_test_subsets, extract_test_labels)  # 先注释掉
    save_h5(out_eval_file, extract_eval_subsets, extract_eval_labels)
    
    # 校验提取后的数据
    # validate_extraction(extract_train_subsets, extract_train_labels, "训练数据", low, high)  # 先注释掉
    # validate_extraction(extract_test_subsets, extract_test_labels, "测试数据", low, high)   # 先注释掉
    validate_extraction(extract_eval_subsets, extract_eval_labels, "评估数据", low, high)

if __name__ == '__main__':
    # 设置提取的标签范围和数据路径
    low = 1
    high = 1000
    base_dir = r'/share/home/202321008879/data/h5data/originnew'
    output_dir = r'/share/home/202321008879/data/h5data/origin1k'
    main(low, high, base_dir, output_dir)
