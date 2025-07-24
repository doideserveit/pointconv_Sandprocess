import os
import csv

def get_particles(file_path):
    """
    读取 csv 文件，返回结构：
    {
        pending_label1: [dict1, dict2, ...], # 每个dict为一个候选颗粒
        pending_label2: [dict3, dict4, ...],
        ...
    }
    """
    particles = {}
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if not row or len(row) < 11:
                continue
            if row[0] == headers[0]:
                continue
            try:
                pending_label = row[0]
                info = {
                    "pending_label": row[0],
                    "candidate_label": row[1],
                    "avg_prob": float(row[2]),
                    "UD": float(row[3]),
                    "UD_ratio": float(row[4]),
                    "weighted_UD": float(row[5]),
                    "LD": float(row[6]),
                    "LD_ratio": float(row[7]),
                    "weighted_LD": float(row[8]),
                    "final_score": float(row[9]),
                    "matched": row[10],
                }
                particles.setdefault(pending_label, []).append(info)
            except ValueError:
                print(f"警告：无法解析行 {row}, 已跳过。")
                continue
    return particles

def get_matched_candidate(candidate_list):
    """返回matched为True的候选颗粒（如有多个，取第一个），没有则返回None"""
    for c in candidate_list:
        if str(c["matched"]) == "True":
            return c
    return None

def compare_particles(csv1, csv2, output_csv, csv1_name="csv1", csv2_name="csv2", all_candidates=False):
    particles1 = get_particles(csv1)
    particles2 = get_particles(csv2)
    set1 = set(particles1.keys())
    set2 = set(particles2.keys())
    common_labels = set1 & set2
    only_in_csv1 = set1 - set2
    only_in_csv2 = set2 - set1
    print(f"CSV1路径： {csv1}")
    print(f"CSV2路径： {csv2}")
    print(f"CSV1 ({csv1_name}) 中独有的标签数量: {len(only_in_csv1)}")
    print(f"CSV2 ({csv2_name}) 中独有的标签数量: {len(only_in_csv2)}")

    differences = []
    diff_pending_labels = set()
    for pending_label in sorted(common_labels):
        cands1 = particles1[pending_label]
        cands2 = particles2[pending_label]
        matched1 = get_matched_candidate(cands1)
        matched2 = get_matched_candidate(cands2)
        # 若两个csv都没有matched，跳过
        if matched1 is None and matched2 is None:
            continue
        # 若匹配结果一致（都无或都matched且candidate_label一致），跳过
        if (matched1 is not None and matched2 is not None and
            matched1["candidate_label"] == matched2["candidate_label"]):
            continue
        diff_pending_labels.add(pending_label)
        if all_candidates:
            # 以所有出现过的candidate_label为主键，合并两个csv的参数
            cand_labels = set([c["candidate_label"] for c in cands1] + [c["candidate_label"] for c in cands2])
        else:
            # 只记录两个csv中的matched candidate_label
            cand_labels = set()
            if matched1 is not None:
                cand_labels.add(matched1["candidate_label"])
            if matched2 is not None:
                cand_labels.add(matched2["candidate_label"])
        c1_dict = {c["candidate_label"]: c for c in cands1}
        c2_dict = {c["candidate_label"]: c for c in cands2}
        for cand in sorted(cand_labels):
            info1 = c1_dict.get(cand)
            info2 = c2_dict.get(cand)
            row = {
                "pending_label": pending_label,
                "candidate_label": cand,
                # CSV1参数
                "csv1_avg_prob": info1["avg_prob"] if info1 else "",
                "csv1_UD": info1["UD"] if info1 else "",
                "csv1_UD_ratio": info1["UD_ratio"] if info1 else "",
                "csv1_weighted_UD": info1["weighted_UD"] if info1 else "",
                "csv1_LD": info1["LD"] if info1 else "",
                "csv1_LD_ratio": info1["LD_ratio"] if info1 else "",
                "csv1_weighted_LD": info1["weighted_LD"] if info1 else "",
                "csv1_final_score": info1["final_score"] if info1 else "",
                "csv1_matched": info1["matched"] if info1 else "",
                # CSV2参数
                "csv2_avg_prob": info2["avg_prob"] if info2 else "",
                "csv2_UD": info2["UD"] if info2 else "",
                "csv2_UD_ratio": info2["UD_ratio"] if info2 else "",
                "csv2_weighted_UD": info2["weighted_UD"] if info2 else "",
                "csv2_LD": info2["LD"] if info2 else "",
                "csv2_LD_ratio": info2["LD_ratio"] if info2 else "",
                "csv2_weighted_LD": info2["weighted_LD"] if info2 else "",
                "csv2_final_score": info2["final_score"] if info2 else "",
                "csv2_matched": info2["matched"] if info2 else "",
            }
            differences.append(row)

    fieldnames = [
        "pending_label", "candidate_label",
        # csv1
        "csv1_avg_prob", "csv1_UD", "csv1_UD_ratio", "csv1_weighted_UD",
        "csv1_LD", "csv1_LD_ratio", "csv1_weighted_LD",
        "csv1_final_score", "csv1_matched",
        # csv2
        "csv2_avg_prob", "csv2_UD", "csv2_UD_ratio", "csv2_weighted_UD",
        "csv2_LD", "csv2_LD_ratio", "csv2_weighted_LD",
        "csv2_final_score", "csv2_matched"
    ]

    with open(output_csv, mode='w', newline='') as f:
        f.write(f"Csv1:,{csv1_name}\n")
        f.write(f"Csv2:,{csv2_name}\n\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(differences)

    print(f"比较完成，结果已保存到 {output_csv}")
    print(f"一共有 {len(diff_pending_labels)} 个pending label的结果不同。")

if __name__ == "__main__":
    info = "halfLD"
    csv1 = "/share/home/202321008879/data/track_results/comparison/load15toload10/half/matchbyanchor_half20.csv"
    csv2 = "/share/home/202321008879/data/track_results/comparison/load15toload10/4d50LD/matchbyanchor_4d50LD10.csv"
    output = "/share/home/202321008879/data/track_results/comparison"
    
    output_csv = os.path.join(output, f"load15_{info}_comparison_results.csv")
    compare_particles(csv1, csv2, output_csv, 'half', info)

