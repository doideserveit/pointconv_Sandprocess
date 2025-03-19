import h5py
import numpy as np

# 定义标签映射关系：key 为原始标签，value 为目标标签
label_mapping = {
    3: 917,
    4: 674,
    6: 760,
    9: 775,
    10: 799,
    5: 723,
    11: 776,
    14: 941,
    2: 635,
    17: 972,
    18: 976,
    15: 811,
    20: 995,
    16: 928,
    7: 751,
    12: 840,
    8: 754,
    19: 796,
    13: 866,
    1: 566
}

def modify_labels(input_file, output_file):
    # 读取输入 h5 文件
    with h5py.File(input_file, 'r') as f_in:
        subsets = f_in['subsets'][...] if 'subsets' in f_in else None
        labels = f_in['labels'][...]
    
    # 根据映射关系修改标签
    new_labels = np.array([label_mapping.get(int(l), l) for l in labels], dtype=np.int32)
    
    # 保存到输出 h5 文件中
    with h5py.File(output_file, 'w') as f_out:
        if subsets is not None:
            f_out.create_dataset('subsets', data=subsets, compression="gzip")
        f_out.create_dataset('labels', data=new_labels, compression="gzip")
    print(f"Modified labels saved to {output_file}")

if __name__ == '__main__':
    input_file = "/share/home/202321008879/data/h5data/load1_selpar/all_eval_fps_subsets_nomapping.h5"
    output_file = "/share/home/202321008879/data/h5data/load1_selpar/all_eval_fps_subsets.h5"
    modify_labels(input_file, output_file)