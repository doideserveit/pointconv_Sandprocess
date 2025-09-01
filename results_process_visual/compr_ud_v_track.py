# 读取采样的100个颗粒的csv文件，获取def_label和predicted_reference_label，并根据def从vol_track的结果中获取匹配结果，并将两者进行对比
import csv
import numpy as np
import os
import argparse

def load_properties(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels = data['label']
    volumes = data['physical_volume(mm)']
    mapping = {}
    for i, lab in enumerate(labels):
        mapping[lab] = {
            'volume': volumes[i]
        }
    return mapping

def load_sample_csv(file_path):
    data = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            def_label = int(row['def_label'])
            ref_label = int(row['predicted_reference_label'])
            data[def_label] = ref_label
    return data

def load_vol_track_results(file_path):
    data = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            def_label = int(row['def_label'])
            ref_label = int(row['ref_label'])
            data[def_label] = ref_label
    return data

def compare_results(sample_data, vol_track_data):
    matches = 0
    total = len(sample_data)
    for def_label, sample_ref in sample_data.items():
        vol_track_ref = vol_track_data.get(def_label)
        if vol_track_ref == sample_ref:
            matches += 1
        else:
            print(f"Def Label: {def_label}, Sample Ref: {sample_ref}, Vol Track Ref: {vol_track_ref}")
    print(f"Total Samples: {total}, Matches: {matches}, Accuracy: {matches/total*100:.2f}%")

def cus_compare(def_labs, sample_refs, vol_track_data, def_mapping, ref_mapping):
    """根据手动给定的def_labels和sample_refs，从properties中获取体积并计算两者的体积误差"""
    match_under_10pct = {}
    match_num_under_10pct = 0
    match_over_10pct = {}
    match_num_over_10pct = 0
    for def_lab, sample_ref in zip(def_labs, sample_refs):
        ref_vol = ref_mapping[sample_ref]['volume']
        def_vol = def_mapping[def_lab]['volume']
        vol_err = abs(def_vol - ref_vol) / def_vol
        vol_track_ref = vol_track_data.get(def_lab)
        if vol_track_ref == sample_ref:
            matched = True
            if vol_err < 0.1:
                match_num_under_10pct += 1
            else:
                match_num_over_10pct += 1
        else:
            matched = False
        if vol_err < 0.1:
            match_under_10pct[def_lab] = {
                    'sample_ref': sample_ref,
                    'vol_err': vol_err,
                    'vol_track_ref': vol_track_ref,
                    'matched': matched,
                }
        else:
            match_over_10pct[def_lab] = {
                    'sample_ref': sample_ref,
                    'vol_err': vol_err,
                    'vol_track_ref': vol_track_ref,
                    'matched': matched,
                }
    print(f'num of matched: {match_num_under_10pct}, num of matches under 10: {len(match_under_10pct)}, over 10%: {len(match_over_10pct)}')
    return match_under_10pct, match_over_10pct

def save_compare_results(match_under_10pct, match_over_10pct, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['def_label', 'sample_ref', 'vol_track_ref', 'vol_err', 'matched'])
        writer.writerow([])
        writer.writerow(['Matches with Volume Error < 10%'])
        for def_lab, info in match_under_10pct.items():
            writer.writerow([def_lab, info['sample_ref'], info['vol_track_ref'], f"{info['vol_err']*100:.2f}%", info['matched']])
        writer.writerow([])
        writer.writerow(['Matches with Volume Error >= 10%'])
        for def_lab, info in match_over_10pct.items():
            writer.writerow([def_lab, info['sample_ref'], info['vol_track_ref'], f"{info['vol_err']*100:.2f}%", info['matched']])
    print(f"Comparison results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="对比采样的100个颗粒的匹配结果与vol_track的结果")
    parser.add_argument('--def_type', required=True, help='变形体系类型，例如 origin, load1, load5, load10, load15')
    parser.add_argument('--ref_type', required=True, help='参考体系类型')
    parser.add_argument('--sample_csv', default=None, help='采样的100个颗粒的csv文件路径')
    parser.add_argument('--vol_track_csv', default=None, help='vol_track的匹配结果csv文件路径')
    parser.add_argument('--ref_npz', default=None, help='参考体系颗粒属性 npz 文件路径')
    parser.add_argument('--def_npz', default=None, help='变形体系颗粒属性 npz 文件路径')
    parser.add_argument('--output_file', default=None, help='对比结果输出csv文件路径')
    args = parser.parse_args()
    data_dir = '/share/home/202321008879/data/'
    if args.sample_csv is None:
        args.sample_csv = os.path.join(data_dir, 'track_results', f'{args.def_type}to{args.ref_type}', f'sampled_100_final_results.csv')
    if args.vol_track_csv is None:
        args.vol_track_csv = os.path.join(data_dir, 'track_results', 'vol_track', f'{args.def_type}to{args.ref_type}', f'match_volume_neighbor_results.csv')
    if args.ref_npz is None:
        args.ref_npz = os.path.join(data_dir, 'sand_propertities', f'{args.ref_type}_ai_particle_properties', f'{args.ref_type}_particle_properties.npz')
    if args.def_npz is None:
        args.def_npz = os.path.join(data_dir, 'sand_propertities', f'{args.def_type}_ai_particle_properties', f'{args.def_type}_particle_properties.npz')
    if args.output_file is None:
        args.output_file = os.path.join(data_dir, 'track_results', f'{args.def_type}to{args.ref_type}', f'compare_sampled_vs_vol_track.csv')
    
    ref_mapping = load_properties(args.ref_npz)
    def_mapping = load_properties(args.def_npz)
    sample_data = load_sample_csv(args.sample_csv)
    vol_track_data = load_vol_track_results(args.vol_track_csv)
    # def_labs = []
    # sample_refs = []
    # match_under_10pct, match_over_10pct = cus_compare(
    #     def_labs, sample_refs, vol_track_data, def_mapping, ref_mapping
    # )
    def_labs = list(sample_data.keys())
    sample_refs = [sample_data[lab] for lab in def_labs]
    match_under_10pct, match_over_10pct = cus_compare(
        def_labs, sample_refs, vol_track_data, def_mapping, ref_mapping
    )
    save_compare_results(match_under_10pct, match_over_10pct, args.output_file)