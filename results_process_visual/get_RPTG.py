# 该脚本用于获取相对颗粒平移梯度relative particle translation gradient (RPTG)数据(未归一化)
import numpy as np
import os
import argparse
import csv

def load_final_results(final_results_file):
    """直接读取final_results.npz，返回def_lab到predicted_reference_label的映射"""
    data = np.load(final_results_file, allow_pickle=True)
    results = data['results'].item()
    mapping = {}
    for def_lab, info in results.items():
        mapping[int(def_lab)] = int(info['predicted_reference_label'])
    return mapping

def load_properties(properties_file):
    """读取properties文件，返回label到centroid的映射"""
    data = np.load(properties_file, allow_pickle=True)
    labels = data['label']
    centroids = data['corrected_centroid(physical)']
    return {int(lab): np.array(centroids[i]) for i, lab in enumerate(labels)}


def compute_displacement_vectors(def2ref, def_props, ref_props):
    """返回每个def_lab的3D位移向量"""
    disp_vecs = {}
    for def_lab, ref_lab in def2ref.items():
        if def_lab in def_props and ref_lab in ref_props:
            disp_vecs[def_lab] = def_props[def_lab] - ref_props[ref_lab]
    return disp_vecs

def get_displacement_vectors_from_results(results_npz_file):
    """针对修正后的final_results,直接从结果文件中获取每个def_lab的3D位移向量"""
    data = np.load(results_npz_file, allow_pickle=True)
    disp_vecs = {}
    for def_lab, info in data['results'].item().items():
        def_centroid = info['corrected_centroid']
        ref_centroid = info['ref_centroid']
        if def_centroid is None or ref_centroid is None:
            print(f"Warning: Label {def_lab} missing centroid(s), skipped.")
            continue
        if np.shape(def_centroid) != (3,) or np.shape(ref_centroid) != (3,):
            print(f"Warning: Label {def_lab} centroid shape is not (3,), skipped.")
            continue
        disp_vecs[int(def_lab)] = np.array(def_centroid) - np.array(ref_centroid)
    return disp_vecs


def read_contacts(contact_file):
    """读取contact.txt，返回每个颗粒的接触颗粒集合"""
    contacts = {}
    with open(contact_file, 'r') as f:
        for line in f:
            a, b = map(int, line.strip().split(','))
            contacts.setdefault(a, set()).add(b)
            contacts.setdefault(b, set()).add(a)
    return contacts

def calculate_rptg(disp_vecs, contacts):
    """计算每个颗粒的RPTG"""
    rptg = {}
    for lab, vec in disp_vecs.items():
        neighbors = contacts.get(lab, [])
        diffs = []
        for nb in neighbors:
            if nb in disp_vecs:
                diff = vec - disp_vecs[nb]
                norm = np.linalg.norm(diff)
                diffs.append(norm)
        if diffs:
            rptg[lab] = np.mean(diffs)
        else:
            rptg[lab] = np.nan  # 无邻居
    return rptg

def save_rptg(rptg, output_file):
    """保存RPTG结果到CSV"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['def_lab', 'RPTG'])
        for lab, val in sorted(rptg.items()):
            writer.writerow([lab, f"{val:.6f}" if not np.isnan(val) else "nan"])

def parse_args():
    parser = argparse.ArgumentParser(description="计算每个颗粒的RPTG")
    parser.add_argument('--def_type', required=True, help='变形体系类型，例如 origin, load1, load5, load10, load15')

    parser.add_argument('--data_dir', default='/share/home/202321008879/data/', help='数据根目录')
    parser.add_argument('--output_dir', default=None, help='输出目录')
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = args.data_dir
    def_type = args.def_type
    ref_list = ['origin', 'load1', 'load5', 'load10', 'load15']
    ref_type = ref_list[ref_list.index(def_type) - 1]

    # 自动补全路径
    final_results_file = os.path.join(data_dir, 'track_results', f'{def_type}to{ref_type}', 'final_results.npz')
    def_props_file = os.path.join(data_dir, 'sand_propertities', f'{def_type}_ai_particle_properties', f'{def_type}_particle_properties.npz')
    ref_props_file = os.path.join(data_dir, 'sand_propertities', f'{ref_type}_ai_particle_properties', f'{ref_type}_particle_properties.npz')
    contact_file = os.path.join(data_dir, 'contact_network', f'{def_type}_contacts.txt')
    output_dir = args.output_dir or os.path.join(data_dir, 'rptg')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{def_type}to{ref_type}_RPTG.csv')

    # 1. 读取匹配结果和质心
    def2ref = load_final_results(final_results_file)
    def_props = load_properties(def_props_file)
    ref_props = load_properties(ref_props_file)
    disp_vecs = compute_displacement_vectors(def2ref, def_props, ref_props)
    # # 直接从修正后的结果文件中获取位移向量
    # disp_vecs = get_displacement_vectors_from_results('/share/home/202321008879/data/track_results/load1toorigin/final_results_corrected.npz')

    # 2. 读取接触信息
    contacts = read_contacts(contact_file)

    # 3. 计算RPTG
    rptg = calculate_rptg(disp_vecs, contacts)

    # 4. 保存结果
    save_rptg(rptg, output_file)
    print(f"RPTG结果已保存到: {output_file}")

if __name__ == '__main__':
    main()

