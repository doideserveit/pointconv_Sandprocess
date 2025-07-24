import argparse
import numpy as np
import os

def load_properties(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels = data['label']
    centroids = data['corrected_centroid(physical)']
    mapping = {}
    for i, lab in enumerate(labels):
        mapping[lab] = np.array(centroids[i])
    return mapping

def main():
    parser = argparse.ArgumentParser(description="计算两个颗粒的质心距离及其为多少倍d50")
    parser.add_argument('--def_type', type=str, default=15, help='属性文件路径')
    parser.add_argument('--l1', type=int, required=True, help='def颗粒的label')
    parser.add_argument('--l2', type=int, required=True, help='ref颗粒的label')
    args = parser.parse_args()
    d50 = 0.72

    type = [0, 1, 5, 10, 15]
    i = type.index(int(args.def_type))
    ref_type = type[i - 1] if i > 0 else type[0]
    strain = type[i]*0.01 - type[i-1]*0.01
    property_dir = '/share/home/202321008879/data/sand_propertities'
    
    
    def_npz = os.path.join(property_dir, f'load{args.def_type}_ai_particle_properties/load{args.def_type}_particle_properties.npz')
    ref_npz = os.path.join(property_dir, f'load{ref_type}_ai_particle_properties/load{ref_type}_particle_properties.npz')
    init_match_file = f"/share/home/202321008879/data/track_results/load{args.def_type}toload{ref_type}/load{args.def_type}toload{ref_type}_initmatch.npz"
    init_match_data = np.load(init_match_file, allow_pickle=True)
    def_sample_height = init_match_data['def_sample_height'].item()
    org_sample_height = 58.57788148034935

    def_mapping = load_properties(def_npz)
    ref_mapping = load_properties(ref_npz)


    centroid1 = def_mapping[args.l1]
    centroid2 = ref_mapping[args.l2]
    relative_z = centroid1[2] / def_sample_height
    print(f"Relative z for label {args.l1}: {relative_z:.6f}")
    displacement = np.array([0, 0, strain * relative_z * org_sample_height])
    mapped_position = centroid1 + displacement
    distance = np.linalg.norm(centroid2 - mapped_position)
    multiple = distance / d50

    print(f"颗粒1 label: {args.l1}, 质心: {centroid1}")
    print(f"颗粒2 label: {args.l2}, 质心: {centroid2}")
    print(f"质心距离: {distance:.6f}")
    print(f"距离为 d50 的 {multiple:.4f} 倍")

    

if __name__ == '__main__':
    main()
