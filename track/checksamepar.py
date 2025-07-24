# 检查相同预测标签的颗粒组，并修正其质心和位移
import csv
import numpy as np

def check_same_predicted_labels(final_results_file, output_file):
    # 读取final_results.csv，按predicted_reference_label分组def_lab
    pred_label_groups = {}
    rows = []
    with open(final_results_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_label = row["predicted_reference_label"]
            def_lab = row["def_lab"]
            if ref_label not in pred_label_groups:
                pred_label_groups[ref_label] = []
            pred_label_groups[ref_label].append(def_lab)
            rows.append(row)

    # 输出有重复的分组
    group_count = 0
    for ref_label, def_labs in pred_label_groups.items():
        if len(def_labs) > 1:
            group_count += 1
    print(f"预测结果相同的颗粒组总数: {group_count}")

    # 保存分组结果到CSV
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Duplicate reference label group", "ref_label", "def_lab1", "def_lab2", "def_lab3", "..."])
        for ref_label, def_labs in pred_label_groups.items():
            if len(def_labs) > 1:
                row = ["", ref_label] + def_labs
                writer.writerow(row)
    print(f"分组结果已保存到 {output_file}")
    return pred_label_groups

def load_properties(properties_file):
    # 加载颗粒属性文件，返回字典：label -> 属性dict
    data = np.load(properties_file, allow_pickle=True)
    labels = data['label']
    centroids = data['corrected_centroid(physical)']
    volumes = data['physical_volume(mm)']
    mapping = {}
    for i, label in enumerate(labels):
        mapping[str(label)] = {
            'centroid': np.array(centroids[i]),
            'volume': float(volumes[i])
        }
    return mapping

def correct_centroid(def_labs, def_mapping):
    # 计算修正质心
    total_volume = 0.0
    weighted_centroid = None
    for lab in def_labs:
        info = def_mapping[lab]
        v = info['volume']
        c = info['centroid']
        if weighted_centroid is None:
            weighted_centroid = v * c
        else:
            weighted_centroid += v * c
        total_volume += v
    corrected = weighted_centroid / total_volume
    return corrected, total_volume

def correct_final_results(final_results_file, def_properties_file, ref_properties_file, corrected_csv, corrected_npz):
    # 读取原始final_results
    with open(final_results_file, mode='r') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    # 加载属性
    def_mapping = load_properties(def_properties_file)
    ref_mapping = load_properties(ref_properties_file)

    # 分组
    pred_label_groups = {}
    for row in rows:
        ref_label = row["predicted_reference_label"]
        def_lab = row["def_lab"]
        if ref_label not in pred_label_groups:
            pred_label_groups[ref_label] = []
        pred_label_groups[ref_label].append(def_lab)

    # 记录修正结果
    corrected_rows = []
    corrected_dict = {}

    for ref_label, def_labs in pred_label_groups.items():
        # 获取参考颗粒质心
        ref_centroid = None
        if ref_label in ref_mapping:
            ref_centroid = ref_mapping[ref_label]['centroid']
        for lab in def_labs:
            row = next(r for r in rows if r["def_lab"] == lab)
            row = row.copy()
            if len(def_labs) > 1:
                corrected_centroid, total_volume = correct_centroid(def_labs, def_mapping)
                if ref_centroid is None:
                    print(f"Warning: 参考标签 {ref_label} 不在参考属性文件中，跳过修正。")
                    corrected_rows.append(row)
                    corrected_dict[lab] = row
                    continue
                displacement = np.linalg.norm(corrected_centroid - ref_centroid)
                row["corrected_centroid_x"] = f"{corrected_centroid[0]:.6f}"
                row["corrected_centroid_y"] = f"{corrected_centroid[1]:.6f}"
                row["corrected_centroid_z"] = f"{corrected_centroid[2]:.6f}"
                row["corrected_displacement"] = f"{displacement:.6f}"
                # 新增参考颗粒质心
                row["ref_centroid_x"] = f"{ref_centroid[0]:.6f}"
                row["ref_centroid_y"] = f"{ref_centroid[1]:.6f}"
                row["ref_centroid_z"] = f"{ref_centroid[2]:.6f}"
                corrected_rows.append(row)
                corrected_dict[lab] = {
                    "predicted_reference_label": row["predicted_reference_label"],
                    "avg_prediction": float(row["avg_prediction"]),
                    "type": row["type"],
                    "displacement": float(row["displacement"]),
                    "corrected_centroid": corrected_centroid,
                    "corrected_displacement": displacement,
                    "ref_centroid": ref_centroid
                }
            else:
                def_centroid = def_mapping[lab]['centroid']
                row["corrected_centroid_x"] = f"{def_centroid[0]:.6f}"
                row["corrected_centroid_y"] = f"{def_centroid[1]:.6f}"
                row["corrected_centroid_z"] = f"{def_centroid[2]:.6f}"
                row["corrected_displacement"] = row["displacement"]
                # 新增参考颗粒质心
                if ref_centroid is not None:
                    row["ref_centroid_x"] = f"{ref_centroid[0]:.6f}"
                    row["ref_centroid_y"] = f"{ref_centroid[1]:.6f}"
                    row["ref_centroid_z"] = f"{ref_centroid[2]:.6f}"
                else:
                    row["ref_centroid_x"] = ""
                    row["ref_centroid_y"] = ""
                    row["ref_centroid_z"] = ""
                corrected_rows.append(row)
                corrected_dict[lab] = {
                    "predicted_reference_label": row["predicted_reference_label"],
                    "avg_prediction": float(row["avg_prediction"]),
                    "type": row["type"],
                    "displacement": float(row["displacement"]),
                    "corrected_centroid": def_centroid,
                    "corrected_displacement": float(row["displacement"]),
                    "ref_centroid": ref_centroid
                }

    # 保存修正后的csv
    fieldnames = list(rows[0].keys()) + [
        "corrected_centroid_x", "corrected_centroid_y", "corrected_centroid_z",
        "corrected_displacement", "ref_centroid_x", "ref_centroid_y", "ref_centroid_z"
    ]
    with open(corrected_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in corrected_rows:
            writer.writerow(row)
    print(f"修正后的final_results已保存到 {corrected_csv}")

    # 保存修正后的npz
    np.savez(corrected_npz, results=corrected_dict)
    print(f"修正后的final_results已保存到 {corrected_npz}")

if __name__ == "__main__":
    def_type = 'load1'  # 变形体系类型
    type_list = ['origin', 'load1', 'load5', 'load10', 'load15']
    ref_type = type_list[type_list.index(def_type) - 1]  # 参考体系类型
    final_results_file = f'/share/home/202321008879/data/track_results/{def_type}to{ref_type}/final_results.csv'
    output_file = f'/share/home/202321008879/data/track_results/{def_type}to{ref_type}/duplicate_predicted_groups.csv'
    def_properties_file = f'/share/home/202321008879/data/sand_propertities/{def_type}_ai_particle_properties/{def_type}_particle_properties.npz'
    ref_properties_file = f'/share/home/202321008879/data/sand_propertities/{ref_type}_ai_particle_properties/{ref_type}_particle_properties.npz'
    corrected_csv = f'/share/home/202321008879/data/track_results/{def_type}to{ref_type}/final_results_corrected.csv'
    corrected_npz = f'/share/home/202321008879/data/track_results/{def_type}to{ref_type}/final_results_corrected.npz'

    groups = check_same_predicted_labels(final_results_file, output_file)
    correct_final_results(final_results_file, def_properties_file, ref_properties_file, corrected_csv, corrected_npz)

