### 这个脚本用于计算颗粒的位移和体积误差，并将结果保存到CSV文件中
import os
import argparse
import numpy as np
import csv

def load_properties(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels = data['label']
    physical_volumes = data['physical_volume(mm)']
    centroids = data['corrected_centroid(physical)'] 
    equivalent_diameters = data['equivalent_diameter(physical)']  
    mapping = {}
    for i, lab in enumerate(labels):
        mapping[lab] = {
            'physical_volume(mm)': physical_volumes[i],
            'corrected_centroid(physical)': np.array(centroids[i]),
            'equivalent_diameter(physical)': equivalent_diameters[i],
        }
    return mapping


def calculate_d50(type='mass', data=None):
    """
    根据累积质量或数量计算占比50%对应的粒径
    """
    if type == 'mass':
        # 计算质量占比50%对应的粒径,假设质量与体积成正比
        total_mass = np.sum(data['physical_volume(mm)'])
        cumulative_mass = np.cumsum(data['physical_volume(mm)'])
        d50_index = np.searchsorted(cumulative_mass, total_mass * 0.5)
        d50 = data['equivalent_diameter(physical)'][d50_index]
    elif type == 'number':
        # 计算数量占比50%对应的粒径
        total_count = len(data['physical_volume(mm)'])
        cumulative_count = np.arange(1, total_count + 1)
        d50_index = np.searchsorted(cumulative_count, total_count * 0.5)
        d50 = data['equivalent_diameter(physical)'][d50_index]
    else:
        raise ValueError("Invalid type. Use 'mass' or 'number'.")
    return d50


def main():
    parser = argparse.ArgumentParser(description="利用预测结果与颗粒属性数据计算体积误差和位移")
    parser.add_argument('--pred_npz', default='/share/home/202321008879/data/track_results/load1toorigin/load1_ai_pred.npz',
                        # /share/home/202321008879/data/track_results/load1_selpartooriginnew_labbotm1k/predictions.npz
                        help="预测结果 npz 文件（包含 deformed_label, predicted_reference_label, avg_prediction）")
    parser.add_argument('--ref_npz', default='/share/home/202321008879/data/sand_propertities/origin_ai/origin_particle_properties.npz', 
                        # /share/home/202321008879/data/sand_propertities/origin_ai/origin_particle_properties.npz
                        # data/sand_propertities/originnew_labbotm1k_particle_properties/originnew_labbotm1k_particle_properties.npz
                        help="参考体系颗粒属性 npz 文件")
    parser.add_argument('--def_npz', default='/share/home/202321008879/data/sand_propertities/load1_ai_particle_properties/load1_particle_properties.npz', 
                        # /share/home/202321008879/data/sand_propertities/load1_ai_particle_properties/load1_particle_properties.npz
                        # /share/home/202321008879/data/sand_propertities/Load1_selpar_particle_properties/Load1_selpar_particle_properties.npz
                        help="变形体系颗粒属性 npz 文件")
    parser.add_argument('--output_dir', default="/share/home/202321008879/data/track_results", help="输出 CSV 路径")
    parser.add_argument("--ref_type", default="origin", help="参考体系类型，例如 origin, load1, load5, load10, load15")
    parser.add_argument("--def_type", default="load1", help="变形体系类型，例如 origin, load1, load5, load10, load15")
    args = parser.parse_args()

    # 定义output文件路径
    output_file = os.path.join(args.output_dir, f"{args.def_type}to{args.ref_type}", f"{args.def_type}to{args.ref_type}_initmatch.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 自动计算应变差
    strain_mapping = {"origin": 0.0, "load1": 0.01, "load5": 0.05, "load10": 0.1, "load15": 0.15, 'originnew_labbotm1k':0.0,'load1_selpar':0.01}
    strain = strain_mapping[args.def_type] - strain_mapping[args.ref_type]
    print(f"Using strain: {strain} (def: {args.def_type} - ref: {args.ref_type})")

    # 加载预测结果
    pred_data = np.load(args.pred_npz, allow_pickle=True)
    predictions = pred_data['results'].item()
    keys = list(predictions.keys())

    # 加载属性数据
    ref_mapping = load_properties(args.ref_npz)
    def_mapping = load_properties(args.def_npz)
    def_sample_height = max([def_mapping[lab]['corrected_centroid(physical)'][2] for lab in def_mapping.keys()])
    
    
    # # 计算 D50（这里采用质量法）
    # ref_npz_data = np.load(args.ref_npz, allow_pickle=True)
    # d50 = calculate_d50(type='mass', data=ref_npz_data)
    # print(f"Computed D50: {d50}")

    threshold = 0.6  # 60%筛选阈值
    d50 = 0.72  # 通过mass和number以及平均等效直径计算得到的D50
    sample_height = 58.57788148034935  # origin样本高度，单位mm
    anchor_particles = {}
    pending_particles = set()

    for def_lab, info in predictions.items():
        # 1.筛选概率小于阈值的结果
        avg_probability = info.get("final_avg_probability", 0)
        # if avg_probability < threshold:
        #     print(f"Skipping {def_lab} due to low average probability: {avg_probability}")
        #     continue
        ref_lab = info.get("final_predicted_label")
        print(f"Processing {def_lab} -> {ref_lab} with avg_probability: {avg_probability}")
        # 2.筛选位移过大的高概率标签
        # 假定：z轴方向按 1% 应变进行线性分布（z=0不动），颗粒位移为 -0.01 * def_centroid[2]（受压方向沿z轴负向），反过来要加上正的位移
        # 通过反推 def 系统中的颗粒位置确定期望参考体系位置
        def_centroid = np.array(def_mapping[def_lab]['corrected_centroid(physical)'])
        relative_z = def_centroid[2] / def_sample_height
        displacement = np.array([0, 0, strain * sample_height * relative_z])
        predicted_position = def_centroid + displacement
        # 取预测标签对应参考体系中的质心
        predicted_ref_centroid = np.array(ref_mapping[ref_lab]['corrected_centroid(physical)'])
        distance = np.linalg.norm(predicted_ref_centroid - predicted_position)
        # 记录参考体系质心位置与预测位置的距离，与2倍D50和4倍D50的大小比较，并作为记录结果为键值
        if distance < 2 * d50:
            distance_range = '<2D50'
        elif distance < 4 * d50 and distance >= 2 * d50:
            distance_range = '2D50-4D50'
        else:
            distance_range = '>4D50'


        # 计算体积误差
        ref_vol = ref_mapping[ref_lab]['physical_volume(mm)']
        def_vol = def_mapping[def_lab]['physical_volume(mm)']
        vol_error = None if ref_vol == 0 else abs(def_vol - ref_vol) / ref_vol * 100
        ref_centroid = np.array(ref_mapping[ref_lab]['corrected_centroid(physical)'])
        def_centroid = np.array(def_mapping[def_lab]['corrected_centroid(physical)'])
        # 显示3个方向上的位移
        x_displacement = def_centroid[0] - ref_centroid[0]
        y_displacement = def_centroid[1] - ref_centroid[1]
        z_displacement = def_centroid[2] - ref_centroid[2]
        displacement_norm = np.linalg.norm(def_centroid - ref_centroid)

        # 判断是否为锚点颗粒或待匹配颗粒
        if avg_probability >= threshold and distance_range == '<2D50' and vol_error is not None and vol_error <= 30:
            anchor_particles[def_lab] = {
                "predicted_reference_label": ref_lab,
                "avg_prediction": avg_probability,
                "volume_pct_error(%)": vol_error,
                "displacement(mm)": displacement_norm,
            }
        else:
            pending_particles.add(def_lab)

    # 保存锚点颗粒和待匹配颗粒到 CSV 文件
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 写入额外参数
        writer.writerow(["sample_height", sample_height])
        writer.writerow(["def_sample_height", def_sample_height])
        writer.writerow([])  # 空行分隔
        # 写入锚点颗粒
        writer.writerow([
            "deformed_label", "predicted_reference_label", "avg_prediction",
            "volume_pct_error(%)", "displacement(mm)",
        ])
        for def_lab, data in anchor_particles.items():
            writer.writerow([
                def_lab, data["predicted_reference_label"], data["avg_prediction"],
                data["volume_pct_error(%)"], data["displacement(mm)"],
            ])
        writer.writerow([])  # 空行分隔
        # 写入待匹配颗粒
        writer.writerow(["deformed_label"])
        for def_lab in pending_particles:
            writer.writerow([def_lab])
        print(f"CSV Results saved to {output_file}")

    # 保存到npz文件
    npz_output = os.path.splitext(output_file)[0] + '.npz'
    np.savez(
        npz_output, 
        anchor_particles=anchor_particles, 
        pending_particles=list(pending_particles),  # 转为列表以便保存
        sample_height=sample_height, 
        def_sample_height=def_sample_height
    )
    print(f"Npz Results saved to {npz_output}")


if __name__ == '__main__':
    main()

# # 计算origin体系中最大的颗粒质心z值作为sample_height
# ref_npz = 'data/sand_propertities/origin_ai/origin_particle_properties.npz'
# ref_mapping = load_properties(ref_npz)
# max_z = max([ref_mapping[lab]['corrected_centroid(physical)'][2] for lab in ref_mapping.keys()])
# print(f"Sample height (max z): {max_z}")


# d50_mass = calculate_d50(type='mass', data=np.load(ref_npz, allow_pickle=True))
# d50_number = calculate_d50(type='number', data=np.load(ref_npz, allow_pickle=True))
# print(f"Mass D50: {d50_mass}")
# print(f"Number D50: {d50_number}")