# 用来简单可视化复杂网络的参数
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(file_path):
    """读取CSV文件并返回数据"""
    data = []
    with open(file_path, 'r') as f:
        csvfile = csv.reader(f)
        param_name = next(csvfile)[1]
        for row in csvfile:
            data.append((int(row[0]), float(row[1])))
    return np.array(data, dtype=[('id', 'i4'), (param_name, 'f4')])


file_path = '/share/home/202321008879/data/contact_network/origin/degree.csv'
data = read_csv(file_path)
degree = data['degree']
# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(degree, bins=50, color='blue', alpha=0.7)
plt.title('Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('/share/home/202321008879/data/contact_network/origin/degree_distribution.png')

