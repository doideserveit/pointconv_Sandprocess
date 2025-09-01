# 指定阶段，分析锚点颗粒体积误差分布
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

def load_properties(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels = data['label']
    physical_volumes = data['physical_volume(mm)']
    mapping = {}
    for i, lab in enumerate(labels):
        mapping[lab] = physical_volumes[i]
    return mapping

def main():
    parser = argparse.ArgumentParser(description="分析锚点颗粒体积误差分布")
    parser.add_argument('--def_type', required=True, help="def阶段")
    parser.add_argument('--ref_type', required=True, help="ref阶段")
    args = parser.parse_args()

    # 读取锚点颗粒
    args.initmatch_npz = f'/share/home/202321008879/data/track_results/{args.def_type}to{args.ref_type}/{args.def_type}to{args.ref_type}_initmatch.npz'
    data = np.load(args.initmatch_npz, allow_pickle=True)
    anchor_particles = data['anchor_particles'].item()

    # 读取体积信息
    args.ref_npz = f'/share/home/202321008879/data/sand_propertities/{args.ref_type}_ai_particle_properties/{args.ref_type}_particle_properties.npz'
    args.def_npz = f'/share/home/202321008879/data/sand_propertities/{args.def_type}_ai_particle_properties/{args.def_type}_particle_properties.npz'
    ref_volumes = load_properties(args.ref_npz)
    def_volumes = load_properties(args.def_npz)

    # 统计体积误差
    error_percentages = []
    particle_errors = []

    for def_lab, info in anchor_particles.items():
        ref_lab = info['predicted_reference_label']
        ref_vol = ref_volumes[ref_lab]
        def_vol = def_volumes[def_lab]
        if ref_vol == 0:
            continue
        error_pct = abs(def_vol - ref_vol) / ref_vol * 100
        error_percentages.append(error_pct)
        particle_errors.append((def_lab, ref_lab, def_vol, ref_vol, error_pct))

    # 统计分布
    bins = [0, 10, 20, 30, 1000]
    counts = np.histogram(error_percentages, bins=bins)[0]
    print("锚点颗粒体积误差分布：")
    print(f"0-10%: {counts[0]}")
    print(f"10-20%: {counts[1]}")
    print(f"20-30%: {counts[2]}")
    # 从20-30%的组中随机挑选3个颗粒出来
    twenty_to_thirty = [(def_lab, ref_lab, def_vol, ref_vol, error_pct) for def_lab, ref_lab, def_vol, ref_vol, error_pct in particle_errors if 20 <= error_pct < 30]
    selected = np.random.choice(len(twenty_to_thirty), size=min(3, len(twenty_to_thirty)), replace=False)
    print("20-30%误差范围内随机挑选的3个颗粒：")
    for idx in selected:
        def_lab, ref_lab, def_vol, ref_vol, error_pct = twenty_to_thirty[idx]
        print(f"Def Label: {def_lab}, Ref Label: {ref_lab}, Def Volume: {def_vol:.2f}, Ref Volume: {ref_vol:.2f}, Error Percentage: {error_pct:.2f}%")
    print(f">30%: {counts[3]}")

if __name__ == '__main__':
    main()
