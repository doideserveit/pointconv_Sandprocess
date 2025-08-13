import os
import numpy as np
import pandas as pd

DATA_DIR = '/share/home/202321008879/data'
STAGE_LIST = ['origin', 'load1', 'load5', 'load10', 'load15']
RPTG_BINS = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

def load_final_results(stage_from, stage_to):
    """读取final_results.npz，返回ref_lab到def_lab的映射（可能一对多）"""
    npz_path = os.path.join(DATA_DIR, 'track_results', f'{stage_to}to{stage_from}', 'final_results.npz')
    data = np.load(npz_path, allow_pickle=True)
    results = data['results'].item()
    ref2def = {}
    for def_lab, info in results.items():
        ref_lab = int(info['predicted_reference_label'])
        ref2def.setdefault(ref_lab, []).append(int(def_lab))
    return ref2def

def load_final_results_reverse(stage_from, stage_to):
    """读取final_results.npz，返回def_lab到ref_lab的映射（一对一）"""
    npz_path = os.path.join(DATA_DIR, 'track_results', f'{stage_to}to{stage_from}', 'final_results.npz')
    data = np.load(npz_path, allow_pickle=True)
    results = data['results'].item()
    return {int(def_lab): int(info['predicted_reference_label']) for def_lab, info in results.items()}

def load_rptg(stage_from, stage_to):
    """读取RPTG csv，返回def_lab到RPTG的映射"""
    csv_path = os.path.join(DATA_DIR, 'rptg', f'{stage_to}to{stage_from}_RPTG.csv')
    df = pd.read_csv(csv_path)
    return dict(zip(df['def_lab'], df['RPTG']))

def group_rptg(val, bins=RPTG_BINS):
    """根据RPTG值分组"""
    for i in range(len(bins)-1):
        if bins[i] <= val < bins[i+1]:
            return f"{bins[i]}-{bins[i+1]}"
    if val >= bins[-2]:
        return f"{bins[-2]}-{bins[-1]}"
    return None

def track_labels_across_stages(input_labels, input_stage):
    """
    输入：input_labels: list[int]，input_stage: str
    输出：dict{stage: set(labels)}
    """
    idx = STAGE_LIST.index(input_stage)
    stage_labels = {input_stage: set(input_labels)}
    # 向前追踪（def_lab->ref_lab，一对一）
    for i in range(idx, 0, -1):
        cur_stage = STAGE_LIST[i]
        prev_stage = STAGE_LIST[i-1]
        def2ref = load_final_results_reverse(prev_stage, cur_stage)
        prev_labels = set()
        for lab in stage_labels[cur_stage]:
            ref_lab = def2ref.get(lab)
            if ref_lab is not None:
                prev_labels.add(ref_lab)
        stage_labels[prev_stage] = prev_labels
    # 向后追踪（ref_lab->def_lab，一对多）
    for i in range(idx, len(STAGE_LIST)-1):
        cur_stage = STAGE_LIST[i]
        next_stage = STAGE_LIST[i+1]
        ref2def = load_final_results(cur_stage, next_stage)
        next_labels = set()
        for lab in stage_labels[cur_stage]:
            next_labels.update(ref2def.get(lab, []))
        stage_labels[next_stage] = next_labels
    return stage_labels

def get_rptg_and_group_for_labels(stage_labels_dict):
    """
    输入：{stage: set(labels)}
    输出：{stage: {label: (rptg, group)}}
    """
    out = {}
    for i, stage in enumerate(STAGE_LIST):
        if stage not in stage_labels_dict:
            continue
        if stage == 'origin':
            rptg_map = load_rptg('origin', 'load1')
        else:
            ref_stage = STAGE_LIST[i-1]
            rptg_map = load_rptg(ref_stage, stage)
        label_info = {}
        for lab in stage_labels_dict[stage]:
            v = rptg_map.get(lab)
            if v is not None:
                try:
                    v = float(v)
                    group = group_rptg(v)
                except:
                    v = None
                    group = None
            else:
                group = None
            label_info[lab] = (v, group)
        out[stage] = label_info
    return out

def find_labels_by_stage_and_group(stage, group):
    """
    输入：stage: str, group: str
    输出：该阶段该分组下所有颗粒的标签list
    """
    i = STAGE_LIST.index(stage)
    if stage == 'origin':
        rptg_map = load_rptg('origin', 'load1')
    else:
        ref_stage = STAGE_LIST[i-1]
        rptg_map = load_rptg(ref_stage, stage)
    out = []
    for lab, v in rptg_map.items():
        try:
            v = float(v)
            g = group_rptg(v)
            if g == group:
                out.append(lab)
        except:
            continue
    return out

def print_track_result(stage_labels):
    print("颗粒在各阶段的标签：")
    for stage in STAGE_LIST:
        labs = stage_labels.get(stage, set())
        print(f"{stage}: {sorted(labs)}")

def print_rptg_and_group(rptg_info):
    print("各阶段颗粒的RPTG值及分组：")
    for stage in STAGE_LIST:
        info = rptg_info.get(stage, {})
        print(f"{stage}:")
        group_counts = {}
        for lab, (v, g) in info.items():
            print(f"  label={lab}, RPTG={v}, group={g}")
            group_counts[g] = group_counts.get(g, 0) + 1
        for g, count in group_counts.items():
            print(f"  分组 {g} 下的颗粒数量: {count}")

def print_rptg_and_group_2stage(rptg_info):
    print("各阶段颗粒的RPTG值及分组：")
    for stage in STAGE_LIST:
        print(f"{stage}:")
        info = rptg_info.get(stage, {})
        group_counts = {}
        for lab, (v, g) in info.items():
            group_counts[g] = group_counts.get(g, 0) + 1
        for g, count in group_counts.items():
            print(f"  分组 {g} 下的颗粒数量: {count}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # 1. 追踪颗粒标签
    p1 = subparsers.add_parser('track', help='追踪颗粒在所有阶段的标签')
    p1.add_argument('--labels', type=int, nargs='+', required=True, help='输入标签')
    p1.add_argument('--stage', type=str, required=True, choices=STAGE_LIST, help='标签所在阶段')

    # 2. 查询颗粒RPTG及分组
    p2 = subparsers.add_parser('rptg', help='查询颗粒在所有阶段的RPTG及分组')
    p2.add_argument('--labels', type=int, nargs='+', required=True)
    p2.add_argument('--stage', type=str, required=True, choices=STAGE_LIST)

    # 3. 反查分组颗粒及其RPTG
    p3 = subparsers.add_parser('group', help='查询某阶段某RPTG分组下颗粒及其在所有阶段的RPTG')
    p3.add_argument('--stage', type=str, required=True, choices=STAGE_LIST)
    p3.add_argument('--group', type=str, required=True, help='RPTG分组,如0-0.1')

    args = parser.parse_args()

    if args.mode == 'track':
        stage_labels = track_labels_across_stages(args.labels, args.stage)
        print_track_result(stage_labels)
    elif args.mode == 'rptg':
        stage_labels = track_labels_across_stages(args.labels, args.stage)
        rptg_info = get_rptg_and_group_for_labels(stage_labels)
        print_rptg_and_group(rptg_info)
    elif args.mode == 'group':
        labs = find_labels_by_stage_and_group(args.stage, args.group)
        # print(f"{args.stage}阶段，分组{args.group}下的颗粒标签: {labs}")
        print(f"{args.stage}阶段")
        stage_labels = track_labels_across_stages(labs, args.stage)
        rptg_info = get_rptg_and_group_for_labels(stage_labels)
        # print_rptg_and_group(rptg_info)
        print_rptg_and_group_2stage(rptg_info)

if __name__ == '__main__':
    main()
