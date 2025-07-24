#!/bin/bash

# 定义基础路径
SCRIPT_PATH="/share/home/202321008879/project/sandprocess/results_process_visual/visualize_displacement.py"

# 定义函数：执行可视化命令
run_visualization() {
  local def_type=$1
  local norml=$2
  local slice_axis=$3
  local color_mode=$4
  local rotate=$5
  local corrected=$6
  local num_slices=$7
  local slice_idx=$8
  
  # 构建命令参数
  cmd="python $SCRIPT_PATH --def_type $def_type --norml $norml --slice_axis $slice_axis --color_mode $color_mode"
  
  # 添加可选参数
  if [ "$rotate" = "true" ]; then
    cmd="$cmd --rotate"
  fi
  
  if [ "$corrected" = "true" ]; then
    cmd="$cmd --corrected"
  fi
  
  if [ -n "$num_slices" ]; then
    cmd="$cmd --num_slices $num_slices"
  fi
  
  if [ -n "$slice_idx" ]; then
    cmd="$cmd --slice_idx $slice_idx"
  fi
  
  echo "正在执行: $cmd"
  eval $cmd
}

# 定义需要处理的参数列表
type_list=("load1" "load5")
norml_list=("actual_disp" "global_norm" "inter_norm")
axis_type=("XY" "YZ" "XZ")
color_modes=("mag" "vector")

# 执行普通模式
echo "===== 执行普通模式 ====="
for def_type in "${type_list[@]}"; do
  echo "=== 处理 def_type: $def_type ==="
  
  # 判断是否需要 corrected 参数
  if [ "$def_type" = "load1" ]; then
    corrected="true"
  else
    corrected="false"
  fi
  
  for norml in "${norml_list[@]}"; do
    echo "  正在处理 norml: $norml"
    
    for axis in "${axis_type[@]}"; do
      echo "    正在处理切片方向: $axis"
      
      # 对每种颜色模式执行命令
      for color_mode in "${color_modes[@]}"; do
        # 所有情况都添加 rotate 参数
        run_visualization "$def_type" "$norml" "$axis" "$color_mode" "true" "$corrected" "" ""
      done
    done
  done
  
  echo "--- $def_type 处理完成 ---"
done

# 执行 num_slices=1 模式
echo "===== 执行 num_slices=1 模式 ====="
for norml in "${norml_list[@]}"; do
  echo "=== 处理 norml: $norml (num_slices=1) ==="
  
  for def_type in "${type_list[@]}"; do
    echo "  正在处理 def_type: $def_type"
    
    # 判断是否需要 corrected 参数
    if [ "$def_type" = "load1" ]; then
      corrected="true"
    else
      corrected="false"
    fi
    
    for axis in "${axis_type[@]}"; do
      echo "    正在处理切片方向: $axis"
      
      # 对每种颜色模式执行命令
      for color_mode in "${color_modes[@]}"; do
        # 所有情况都添加 rotate 参数和 num_slices=1
        run_visualization "$def_type" "$norml" "$axis" "$color_mode" "true" "$corrected" "1" ""
      done
    done
  done
  
  echo "--- $norml (num_slices=1) 处理完成 ---"
done