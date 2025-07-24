#!/bin/bash

python init_particle_track.py \
  --pred_npz /share/home/202321008879/data/track_results/load5toorigin/load5_ai_pred.npz \
  --ref_npz /share/home/202321008879/data/sand_propertities/origin_ai_particle_properties/origin_particle_properties.npz \
  --def_npz /share/home/202321008879/data/sand_propertities/load5_ai_particle_properties/load5_particle_properties.npz \
  --ref_type origin \
  --def_type load5

python init_particle_track.py \
  --pred_npz /share/home/202321008879/data/track_results/load10toorigin/load10_ai_pred.npz \
  --ref_npz /share/home/202321008879/data/sand_propertities/origin_ai_particle_properties/origin_particle_properties.npz \
  --def_npz /share/home/202321008879/data/sand_propertities/load10_ai_particle_properties/load10_particle_properties.npz \
  --ref_type origin \
  --def_type load10

echo "全部任务完成"


