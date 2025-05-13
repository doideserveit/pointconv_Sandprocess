import os
import numpy as np
from tifffile import imread
import csv
from joblib import Parallel, delayed

def get_particle_centroid(data, label):
    """
    计算给定label颗粒的质心坐标
    """
    indices = np.nonzero(data == label)
    if indices[0].size == 0:
        raise ValueError(f"Label {label} not found in data.")
    centroid = np.array([np.mean(idx) for idx in indices])
    return centroid

def get_particle_volume(data, label):
    """
    计算给定label颗粒的体积（体素数量）以及其物理体积（单位：mm³）
    """
    voxel_count = np.sum(data == label)
    voxel_size = 0.033321  # 单位: mm
    physical_volume = voxel_count * (voxel_size ** 3)
    return voxel_count, physical_volume

def equivalent_diameter(volume):
    """
    计算等效直径（单位：mm），根据体积(体素或物理)计算。
    """
    if volume <= 0:
        raise ValueError("Volume must be positive.")
    diameter = (3 * volume / (4 * np.pi)) ** (1/3)
    return diameter

def update_coords(coords, params):
    """
    修改质心坐标，根据传入的参数：
      - 将x、y轴的原点平移到 params["offset"]（lattice index），
      - z轴以 params["z_origin"] 切片为原点；
    返回修正后的index坐标和对应的物理坐标（乘以 params["voxel_size"]）。
    """
    # voxel_size = params.get("voxel_size", 0.033321)
    # offset = params.get("offset", [0, 0])
    # corrected_index = np.array([coords[0] - offset[0],
    #                               coords[1] - offset[1],
    #                               coords[2]])
    # corrected_physical = corrected_index * voxel_size
    # return corrected_index, corrected_physical
    voxel_size = params.get("voxel_size", 0.033321)
    offset = params.get("offset", [0, 0])
    zoffset = params.get("zorigin", 0)
    corrected_index = np.array([coords[0] - offset[0],
                                  coords[1] - offset[1],
                                  coords[2] + zoffset,] )
    corrected_physical = corrected_index * voxel_size
    return corrected_index, corrected_physical


def save_properties_npz(output_path, properties):
    """
    以 npz 格式保存所有属性，无需手动更新参数。
    """
    np.savez_compressed(output_path, **properties)
    print(f"Properties saved to {output_path}")

def flatten_value(val):
    """
    将值展平为列表。如果 val 为标量则返回［val］，否则展平为列表。
    """
    if np.isscalar(val):
        return [val]
    else:
        return np.array(val).ravel().tolist()

def save_properties_csv(output_path, properties):
    """
    自动生成CSV文件，将 properties 中所有键的值展平，无需硬编码字段。
    """
    # 假设 properties 中每个键的列表长度相同
    num_rows = len(next(iter(properties.values())))
    headers = []
    # 生成标题：若首元素展平后长度为1，则标题为 key，否则为 key_0, key_1, ...
    col_config = {}  # 记录每个 key 是否需要展平
    for key, vals in properties.items():
        sample = flatten_value(vals[0])
        if len(sample) == 1:
            headers.append(key)
            col_config[key] = 1
        else:
            col_config[key] = len(sample)
            for i in range(len(sample)):
                headers.append(f"{key}_{i}")
    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for i in range(num_rows):
            row = []
            for key, vals in properties.items():
                flat = flatten_value(vals[i])
                if col_config[key] == 1:
                    row.append(flat[0])
                else:
                    row.extend(flat)
            writer.writerow(row)
    print(f"Properties CSV saved to {output_path}")

def process_label_parallel(label, data, params):
    """
    并行处理单个标签，返回 (label, centroid, voxel_volume, physical_volume, 
    corrected_index, corrected_physical) 元组；
    centroid为原始质心（lattice index）。
    """
    try:
        centroid = get_particle_centroid(data, label)
        voxel_cnt, physical_vol = get_particle_volume(data, label)
        corrected_index, corrected_physical = update_coords(centroid, params)
        voxel_eq_diameter = equivalent_diameter(voxel_cnt)
        physical_eq_diameter = equivalent_diameter(physical_vol)
        print(f"Label {label}: voxel_volume = {voxel_cnt}, physical_volume = {physical_vol}, corrected_index = {corrected_index}, "
                f'equivalent_diameter(mm) = {physical_eq_diameter}') 
        return (label, centroid, voxel_cnt, physical_vol, corrected_index, corrected_physical, voxel_eq_diameter, physical_eq_diameter)
    except Exception as e:
        print(f"Processing label {label} failed: {e}")
        return (label, None, None, None, None, None, None, None)

def main(tif_file, output_dir, tif_name, correction_params):
    """
    主函数，读取 tif 文件，计算颗粒属性并保存（并行处理标签以充分利用多核）
    correction_params: 用于坐标修正的参数字典，将直接传递给 update_coords。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取 tif 数据，原始数据格式为 (z,y,x)，转换为 (x,y,z)
    data = imread(tif_file)
    print(f"原始数据维数 (z,y,x): {data.shape}")
    data = np.transpose(data, (2, 1, 0)).copy()
    print(f"转换后数据维数 (x,y,z): {data.shape}")
    
    unique_labels = np.unique(data)
    particle_labels = unique_labels[unique_labels > 0]
    print(f"共找到 {len(particle_labels)} 个颗粒")
    
    # 并行处理所有标签，充分利用多核
    results = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(process_label_parallel)(label, data, correction_params) for label in particle_labels
    )
    
    # 在 properties 字典中分别保存原始质心、以及修正后的index和物理坐标值
    properties = {
        'label': [],
        'centroid(index)': [],
        'voxel_volume': [],
        'physical_volume(mm)': [],
        'corrected_centroid(index)': [],
        'corrected_centroid(physical)': [],
        'equivalent_diameter(voxel)': [],
        'equivalent_diameter(physical)': [],
    }
    
    for res in results:
        label, centroid, voxel_cnt, physical_vol, corrected_index, corrected_physical, voxel_eq_diameter, physical_eq_diameter = res
        if centroid is None:
            continue
        properties['label'].append(label)
        properties['centroid(index)'].append(centroid)
        properties['voxel_volume'].append(voxel_cnt)
        properties['physical_volume(mm)'].append(physical_vol)
        properties['corrected_centroid(index)'].append(corrected_index)
        properties['corrected_centroid(physical)'].append(corrected_physical)
        properties['equivalent_diameter(voxel)'].append(voxel_eq_diameter)
        properties['equivalent_diameter(physical)'].append(physical_eq_diameter)
    
    # 保存为 npz 格式
    npz_output = os.path.join(output_dir, tif_name + '_particle_properties.npz')
    save_properties_npz(npz_output, properties)

    # 保存为 CSV 格式
    csv_output = os.path.join(output_dir, tif_name + '_particle_properties.csv')
    save_properties_csv(csv_output, properties)

if __name__ == '__main__':
    # all_tif_dir = '/share/home/202321008879/data/sandlabel'
    # all_output_dir = '/share/home/202321008879/data/sand_propertities'
   # 使用包含tif名称和对应修正坐标参数的字典列表
    # tif_list = [
    #     # {"name": "origin", "offset": [647.373, 644.098], "voxel_size": 0.033321},
    #     # 可在此添加更多字典，例如:
    #     {"name": "load1", "offset": [652.145, 651.986], "voxel_size": 0.033321},
    #     # {"name": "load5", "offset": [687.690, 680.105], "voxel_size": 0.033321},
    #     # {"name": "load10", "offset": [734.805, 733.713], "voxel_size": 0.033321},
    #     # {"name": "load15", "offset": [798.219, 786.444], "voxel_size": 0.033321},
    # ]
    # for tif_dict in tif_list:
    #     tif_file = os.path.join(all_tif_dir, tif_dict["name"] + '_ai_label.tif')
    #     output_dir = os.path.join(all_output_dir, tif_dict["name"] + '_ai_particle_properties')
    #     main(tif_file, output_dir, tif_dict["name"], correction_params=tif_dict)
    #     print(f"Finished processing {tif_dict['name']}_ai_label.tif")

    testtif_dir= '/share/home/202321008879/data/sandlabel/old'
    base_output_dir = '/share/home/202321008879/data/sand_propertities/'
    tif_list = [{"name": "originnew_labbotm1k", "offset": [647.373, 644.098], 'zorigin':1661, "voxel_size": 0.033321},
                {"name": "Load1_selpar", "offset": [652.145, 651.986], 'zorigin': 1656, "voxel_size": 0.033321},
                ]
    for tif_dict in tif_list:
        tif_file = os.path.join(testtif_dir, tif_dict["name"] + '.label.tif')
        output_dir = os.path.join(base_output_dir, tif_dict["name"] + '_particle_properties')
        main(tif_file, output_dir, tif_dict["name"], correction_params=tif_dict)
        print(f"Finished processing {tif_dict['name']}_ai_label.tif")
