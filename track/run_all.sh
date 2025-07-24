#!/bin/bash

python nnpred.py --checkpoint /share/home/202321008879/experiment/originai_classes25274_points1200_2025-05-16_10-18/checkpoints/originai-0.999755-0100.pth \
  --data_root /share/home/202321008879/data/h5data/load1_ai \
  --output_file /share/home/202321008879/data/track_results/load1toorigin/load1_ai_pred.csv

python init_particle_track.py \
  --pred_npz /share/home/202321008879/data/track_results/load1toorigin/load1_ai_pred.npz \
  --ref_npz /share/home/202321008879/data/sand_propertities/origin_ai_particle_properties/origin_particle_properties.npz \
  --def_npz /share/home/202321008879/data/sand_propertities/load1_ai_particle_properties/load1_particle_properties.npz

python matchbyanchor.py \
  --ref_npz /share/home/202321008879/data/sand_propertities/origin_ai_particle_properties/origin_particle_properties.npz \
  --def_npz /share/home/202321008879/data/sand_propertities/load1_ai_particle_properties/load1_particle_properties.npz \
  --pred_npz /share/home/202321008879/data/track_results/load1toorigin/load1_ai_pred.npz \
  --output_dir /share/home/202321008879/data/track_results/load1toorigin/matchbyanchor \
  --init_match_file /share/home/202321008879/data/track_results/load1toorigin/load1toorigin_initmatch.npz

python nnpred.py --checkpoint /share/home/202321008879/experiment/originai_classes25274_points1200_2025-05-16_10-18/checkpoints/originai-0.999755-0100.pth \
  --data_root /share/home/202321008879/data/h5data/load5_ai \
  --output_file /share/home/202321008879/data/track_results/load5toorigin/load5_ai_pred.csv

echo "全部任务完成"


