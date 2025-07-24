#!/bin/bash

# 定义基础路径
SCRIPT_PATH="/share/home/202321008879/project/sandprocess/results_process_visual/visualize_rptg.py"
DATA_DIR="/share/home/202321008879/data/"

# 定义需要处理的参数列表
type_list=("load1" "load5" "load10" "load15")
axis_list=("XY" "YZ" "XZ")

# 循环1：每个load每个axis生成10张切片
echo "===== 开始循环1：每个load每个axis生成10张切片 ====="
for def_type in "${type_list[@]}"; do
    echo "=== 处理 def_type: $def_type (10张切片) ==="
    
    for axis in "${axis_list[@]}"; do
        echo "  正在处理切片方向: $axis"
        
        # 执行命令：10张切片，rotate=True
        python "$SCRIPT_PATH" \
            --def_type "$def_type" \
            --slice_axis "$axis" \
            --num_slices 10 \
            --data_dir "$DATA_DIR" \
            --rotate
    done
    
    echo "--- $def_type (10张切片) 处理完成 ---"
done

# 循环2：每个load每个axis生成1张切片
echo "===== 开始循环2：每个load每个axis生成1张切片 ====="
for def_type in "${type_list[@]}"; do
    echo "=== 处理 def_type: $def_type (1张切片) ==="
    
    for axis in "${axis_list[@]}"; do
        echo "  正在处理切片方向: $axis"
        
        # 执行命令：1张切片，slice_idx=None，rotate=True
        python "$SCRIPT_PATH" \
            --def_type "$def_type" \
            --slice_axis "$axis" \
            --num_slices 1 \
            --data_dir "$DATA_DIR" \
            --rotate
    done
    
    echo "--- $def_type (1张切片) 处理完成 ---"
done