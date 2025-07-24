### 这个脚本用于计算颗粒的位移和体积误差，并将结果保存到CSV文件中
import os
import argparse
import numpy as np
import csv
import h5py
from joblib import Parallel, delayed

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


def get_prediction_for_particle(h5_path, def_lab):
    """
    按需读取单个颗粒的预测信息
    """
    with h5py.File(h5_path, 'r') as h5f:
        g = h5f['particles'][str(def_lab)]
        subsets = []
        for j in range(len(g["subset_predicted_label"])):
            subsets.append({
                "predicted_label": int(g["subset_predicted_label"][j]),
                "probability": g["subset_probability"][j].tolist()
            })
        prediction = {
            "subsets": subsets,
            "final_predicted_label": int(g.attrs["final_predicted_label"]),
            "final_avg_probability": float(g.attrs["final_avg_probability"])
        }
    return prediction


def main():
    parser = argparse.ArgumentParser(description="利用预测结果与颗粒属性数据计算体积误差和位移")
    parser.add_argument("--ref_type", required=True, help="参考体系类型，例如 origin, load1, load5, load10, load15")
    parser.add_argument("--def_type", required=True, help="变形体系类型，例如 origin, load1, load5, load10, load15")
    parser.add_argument('--multiple', default=2, type = int, help="计算搜索半径时的倍数，如搜索半径为2倍D50")

    parser.add_argument('--pred_h5', default=None, help="预测结果 h5 文件（由 nnpred.py 生成，包含每个颗粒的预测信息）")
    parser.add_argument('--ref_npz', default=None, help="参考体系颗粒属性 npz 文件")
    parser.add_argument('--def_npz', default=None, help="变形体系颗粒属性 npz 文件")
    parser.add_argument('--output_dir', default="/share/home/202321008879/data/track_results", help="输出 CSV 路径")
    args = parser.parse_args()

    # 自动补全文件路径
    if args.pred_h5 is None:
        args.pred_h5 = f"/share/home/202321008879/data/track_results/{args.def_type}to{args.ref_type}/{args.def_type}_ai_pred.h5"
    if args.ref_npz is None:
        args.ref_npz = f"/share/home/202321008879/data/sand_propertities/{args.ref_type}_ai_particle_properties/{args.ref_type}_particle_properties.npz"
    if args.def_npz is None:
        args.def_npz = f"/share/home/202321008879/data/sand_propertities/{args.def_type}_ai_particle_properties/{args.def_type}_particle_properties.npz"

    # 定义output文件路径
    output_file = os.path.join(args.output_dir, f"{args.def_type}to{args.ref_type}", f"{args.def_type}to{args.ref_type}_initmatch.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 自动计算应变差
    strain_mapping = {"origin": 0.0, "load1": 0.01, "load5": 0.05, "load10": 0.1, "load15": 0.15, 'originnew_labbotm1k':0.0,'load1_selpar':0.01}
    strain = strain_mapping[args.def_type] - strain_mapping[args.ref_type]
    print(f"Using strain: {strain} (def: {args.def_type} - ref: {args.ref_type})")

    # 加载预测结果（只传递h5路径，不预加载全部数据）
    pred_h5_path = args.pred_h5
    # 获取所有def标签（通过def_mapping）
    def_labels = list(load_properties(args.def_npz).keys())
    
    # 加载属性数据
    ref_mapping = load_properties(args.ref_npz)
    def_mapping = load_properties(args.def_npz)
    def_sample_height = max([def_mapping[lab]['corrected_centroid(physical)'][2] for lab in def_mapping.keys()])
    
    threshold = 0.6  # 60%筛选阈值
    d50 = 0.72  # 通过mass和number以及平均等效直径计算得到的D50
    org_sample_height = 58.57788148034935
    ref_sample_height = max([ref_mapping[lab]['corrected_centroid(physical)'][2] for lab in ref_mapping.keys()]) # load1=57.834131855788314
    print(f"Ref sample height (max z): {ref_sample_height}")
    print(f'The multiple for distance range is set to: {args.multiple}d50')
    
    def process_particle(def_lab, multiple= args.multiple ):
        # 按需读取预测信息
        info = get_prediction_for_particle(pred_h5_path, def_lab)
        avg_probability = info.get("final_avg_probability", 0)
        ref_lab = info.get("final_predicted_label")
        def_centroid = np.array(def_mapping[def_lab]['corrected_centroid(physical)'])
        relative_z = def_centroid[2] / def_sample_height
        displacement = np.array([0, 0, strain * org_sample_height * relative_z])
        predicted_position = def_centroid + displacement
        predicted_ref_centroid = np.array(ref_mapping[ref_lab]['corrected_centroid(physical)'])
        distance = np.linalg.norm(predicted_ref_centroid - predicted_position)
        if distance < multiple * d50:
            distance_range = '<2D50'
        elif distance < 2* multiple * d50:
            distance_range = '2D50-4D50'
        else:
            distance_range = '>4D50'
        
        ref_vol = ref_mapping[ref_lab]['physical_volume(mm)']
        def_vol = def_mapping[def_lab]['physical_volume(mm)']
        vol_error = None if ref_vol == 0 else abs(def_vol - ref_vol) / ref_vol * 100
        ref_centroid = np.array(ref_mapping[ref_lab]['corrected_centroid(physical)'])
        def_centroid = np.array(def_mapping[def_lab]['corrected_centroid(physical)'])
        x_displacement = def_centroid[0] - ref_centroid[0]
        y_displacement = def_centroid[1] - ref_centroid[1]
        z_displacement = def_centroid[2] - ref_centroid[2]
        displacement_norm = np.linalg.norm(def_centroid - ref_centroid)
        # 判断是否为锚点颗粒或待匹配颗粒
        if avg_probability >= threshold and distance_range == '<2D50' and vol_error is not None and vol_error <= 30:
            return ('anchor', def_lab, {
                "predicted_reference_label": ref_lab,
                "avg_prediction": avg_probability,
                "volume_pct_error(%)": vol_error,
                "displacement(mm)": displacement_norm,
            })
        else:
            return ('pending', def_lab, None)

    # 多进程并行处理颗粒
    results = Parallel(n_jobs=-1, batch_size=32, prefer="threads")(
        delayed(process_particle)(def_lab) for def_lab in def_mapping.keys()
    )

    anchor_particles = {}
    pending_particles = set()
    for status, def_lab, data in results:
        if status == 'anchor':
            anchor_particles[def_lab] = data
        else:
            pending_particles.add(def_lab)

    # 保存锚点颗粒和待匹配颗粒到 CSV 文件
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ref_sample_height", ref_sample_height])
        writer.writerow(["def_sample_height", def_sample_height])
        writer.writerow([])  # 空行分隔
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
        writer.writerow(["deformed_label"])
        for def_lab in pending_particles:
            writer.writerow([def_lab])
        print(f"CSV Results saved to {output_file}")

    # 保存到npz文件
    npz_output = os.path.splitext(output_file)[0] + '.npz'
    np.savez(
        npz_output, 
        anchor_particles=anchor_particles, 
        pending_particles=list(pending_particles), 
        ref_sample_height=ref_sample_height, 
        def_sample_height=def_sample_height
    )
    print(f"Npz Results saved to {npz_output}")


if __name__ == '__main__':
    main()