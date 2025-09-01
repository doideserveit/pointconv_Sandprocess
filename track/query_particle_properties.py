import argparse
import numpy as np
import os

def load_properties(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    properties = {}
    for key in data.files:
        properties[key] = data[key]
    return properties

def query_properties(properties, labels, param):
    label_arr = properties['label']
    idx_map = {lab: i for i, lab in enumerate(label_arr)}
    results = []
    for lab in labels:
        if lab not in idx_map:
            results.append((lab, None))
        else:
            val = properties[param][idx_map[lab]]
            results.append((lab, val))
    return results

def main():
    parser = argparse.ArgumentParser(description="查询颗粒属性npz文件中指定标签的参数")
    parser.add_argument('--stage', required=True, help='阶段名，例如 origin, load1, load5, load10, load15')
    parser.add_argument('--param', required=True, help='要查询的参数名，如 physical_volume(mm), voxel_volume, centroid(index), corrected_centroid(physical) 等')
    parser.add_argument('--labels', type=int, nargs='+', required=True, help='要查询的颗粒标签（可多个）')
    parser.add_argument('--npz_path', default=None, help='属性npz文件路径')
    args = parser.parse_args()

    data_dir = '/share/home/202321008879/data/sand_propertities'
    if args.npz_path is None:
        args.npz_path = os.path.join(data_dir, f'{args.stage}_ai_particle_properties', f'{args.stage}_particle_properties.npz')

    properties = load_properties(args.npz_path)
    if args.param not in properties:
        print(f"参数 {args.param} 不在属性文件中。可选参数：{list(properties.keys())}")
        return

    results = query_properties(properties, args.labels, args.param)
    for lab, val in results:
        print(f"label={lab}  {args.param}={val}")

if __name__ == '__main__':
    main()
