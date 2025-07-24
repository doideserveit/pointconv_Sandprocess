# 该脚本读取csv，并将其两列以阶段重命名，然后汇总到一个csv中
import os
import pandas as pd

def merge_network_params(
    stages, 
    param_name, 
    input_dir_template, 
    output_csv
):
    dfs = []
    for stage in stages:
        csv_path = os.path.join(input_dir_template.format(stage=stage), f"{param_name}.csv")
        df = pd.read_csv(csv_path)
        # 重命名参数列
        df = df.rename(columns={param_name: f"{stage}_{param_name}"})
        dfs.append(df)
    
    # 按 node 合并所有阶段
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="node", how="outer")
    
    merged_df.to_csv(output_csv, index=False)
    print(f"已合并保存到: {output_csv}")

if __name__ == "__main__":
    # 示例：你可以根据实际情况修改这些参数
    stages = ["origin", "load1", "load5", "load10", "load15"]
    param_name = "closeness"  # 例如 degree, clustering, betweenness 等
    input_dir_template = "/share/home/202321008879/data/contact_network/{stage}"
    output_csv = f"/share/home/202321008879/data/contact_network/merged_{param_name}.csv"
    
    merge_network_params(stages, param_name, input_dir_template, output_csv)

