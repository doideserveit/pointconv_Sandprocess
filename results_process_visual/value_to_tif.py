# 这个脚本将对应的数值赋给标签tif文件，并生成新的tif文件。
import numpy as np
import tifffile as tiff
import argparse
import os
import csv

def load_rptg_csv(csv_file):
    """读取RPTG csv，返回label到RPTG值的映射"""
    mapping = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = int(row['def_lab'])
            val = row['RPTG']
            mapping[lab] = float(val) if val != 'nan' else np.nan
    return mapping

def load_properties(npz_path):
    """
    加载颗粒属性文件，返回 {label: centroid}
    """
    data = np.load(npz_path, allow_pickle=True)
    labels = data['label']
    centroids = data['corrected_centroid(physical)']
    mapping = {}
    for i, lab in enumerate(labels):
        mapping[int(lab)] = np.array(centroids[i])
    return mapping

def compute_displacement_mapping_from_results_and_properties(results_npz_file, def_properties_npz, ref_properties_npz):
    """
    从results_npz_file中获取def label和其对应的ref label，
    然后分别从def/ref的properties中获取质心，计算位移（def质心 - ref质心）。
    返回: {def_label: displacement (3,)}
    """
    def_mapping = load_properties(def_properties_npz)
    ref_mapping = load_properties(ref_properties_npz)
    data = np.load(results_npz_file, allow_pickle=True)
    results = data['results'].item()
    displacement_mapping = {}
    for def_label, info in results.items():
        ref_label = info.get('predicted_reference_label', None)
        if ref_label is None:
            print(f"Warning: def_label {def_label} has no 'predicted_reference_label' in results, skipped.")
            continue
        def_label = int(def_label)
        ref_label = int(ref_label)
        if def_label not in def_mapping:
            print(f"Warning: def_label {def_label} not found in def properties, skipped.")
            continue
        if ref_label not in ref_mapping:
            print(f"Warning: ref_label {ref_label} not found in ref properties, skipped.")
            continue
        def_centroid = def_mapping[def_label]
        ref_centroid = ref_mapping[ref_label]
        disp = np.array(def_centroid) - np.array(ref_centroid)
        if disp.shape != (3,):
            print(f"Warning: Label {def_label} displacement shape {disp.shape} is not (3,). Skipped.")
            continue
        displacement_mapping[def_label] = disp
    if len(displacement_mapping) == 0:
        print("Error: No valid displacement mapping found. Please check your results file and properties files.")
    return displacement_mapping

def get_displacement_mapping_from_corrected_results(results_npz_file):
    """
    这个函数是专门为修正后的final_results设计的。
    从corrected_results_npz_file中获取def label和其对应的ref label，
    同时在里面中获取质心，计算位移（def质心 - ref质心）。
    返回: {def_label: displacement (3,)}
    """
    data = np.load(results_npz_file, allow_pickle=True)
    displacement_mapping = {}
    for lab, info in data['results'].item().items():
        def_centroid = info['corrected_centroid']
        ref_centroid = info['ref_centroid']
        if def_centroid is None or ref_centroid is None:
            print(f"Warning: Label {lab} missing centroid(s), skipped.")
            continue
        if np.shape(def_centroid) != (3,) or np.shape(ref_centroid) != (3,):
            print(f"Warning: Label {lab} centroid shape is not (3,), skipped.")
            continue
        disp = np.array(def_centroid) - np.array(ref_centroid)
        displacement_mapping[int(lab)] = disp
    return displacement_mapping

def assign_value_to_tif(labels_3d, value_mapping, default_value=0.0):
    """
    用查找表将value_mapping赋值到labels_3d，返回value体素图
    """
    max_label = labels_3d.max()
    lut = np.full(max_label + 1, default_value, dtype=np.float32)
    for lab in range(max_label + 1):
        val = value_mapping.get(lab, default_value)
        if np.isnan(val):
            lut[lab] = default_value
        else:
            lut[lab] = val
    return lut[labels_3d]

def parse_args():
    parser = argparse.ArgumentParser(description="将RPTG值或位移赋给label tif，生成新的tif文件")
    parser.add_argument('--def_type', required=True, help='变形体系类型，例如 origin, load1, load5, load10, load15')
    parser.add_argument('--data_dir', default='/share/home/202321008879/data/', help='数据根目录')
    parser.add_argument('--output_dir', default='/share/home/202321008879/data/visualization', help='输出tif目录')
    parser.add_argument('--value_type', choices=['rptg', 'disp'], default='rptg', help='赋值类型: rptg 或 disp')
    # parser.add_argument('--results_npz', help='当value_type=disp时，指定results npz文件路径')
    # parser.add_argument('--def_properties_npz', help='当value_type=disp时，指定def properties npz路径')
    # parser.add_argument('--ref_properties_npz', help='当value_type=disp时，指定ref properties npz路径')
    return parser.parse_args()

def main():
    args = parse_args()
    def_type = args.def_type
    ref_list = ['origin', 'load1', 'load5', 'load10', 'load15']
    ref_idx = ref_list.index(def_type) - 1
    ref_type = ref_list[ref_idx]
    tif_file = os.path.join(args.data_dir, 'sandlabel', f'{def_type}_ai_label_transformed.tif')
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载label数据
    labels_3d = tiff.imread(tif_file)
    labels_3d = np.transpose(labels_3d, (2, 1, 0)).copy()

    if args.value_type == 'rptg':
        rptg_csv = os.path.join(args.data_dir, 'rptg', f'{def_type}to{ref_type}_RPTG.csv')
        output_tif = os.path.join(args.output_dir, f'rptg/{def_type}_rptg_rotate.tif')
        rptg_mapping = load_rptg_csv(rptg_csv)
        # ==== 在这里修改RPTG值（如归一化、截断） ====
        org_sample_height = 58.57788148034935
        global_disp = 0.15 * org_sample_height
        for lab, val in rptg_mapping.items():
            rptg_mapping[lab] = val / global_disp  # 归一化
            if not np.isnan(rptg_mapping[lab]):
                rptg_mapping[lab] = min(rptg_mapping[lab], 0.06)
        # ==========================================
        value_vol = assign_value_to_tif(labels_3d, rptg_mapping, default_value=0.0)
    elif args.value_type == 'disp':
        args.results_npz = f"/share/home/202321008879/data/track_results/{args.def_type}to{ref_type}/final_results.npz"
        args.def_properties_npz = f"/share/home/202321008879/data/sand_propertities/{args.def_type}_ai_particle_properties/{args.def_type}_particle_properties.npz"
        args.ref_properties_npz = f"/share/home/202321008879/data/sand_propertities/{ref_type}_ai_particle_properties/{ref_type}_particle_properties.npz"
        if args.def_type == 'load1':
            args.results_npz = f"/share/home/202321008879/data/track_results/{args.def_type}to{ref_type}/final_results_corrected.npz"
            disp_mapping = get_displacement_mapping_from_corrected_results(args.results_npz)
        else:
            disp_mapping = compute_displacement_mapping_from_results_and_properties(
                args.results_npz, args.def_properties_npz, args.ref_properties_npz)
        # 取模长
        disp_norm_mapping = {lab: float(np.linalg.norm(vec)) for lab, vec in disp_mapping.items()}
        output_tif = os.path.join(args.output_dir, f'{def_type}_disp_rotate.tif')
        value_vol = assign_value_to_tif(labels_3d, disp_norm_mapping, default_value=0.0)
    else:
        raise ValueError(f"不支持的value_type: {args.value_type}")

    # 保存tif
    value_vol = np.transpose(value_vol, (2, 1, 0))  # 转回原始轴顺序
    tiff.imwrite(output_tif, value_vol, compression='zlib')
    print(f"Saved: {output_tif}")

if __name__ == '__main__':
    main()
