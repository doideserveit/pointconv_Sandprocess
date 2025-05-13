import numpy as np

npz_file = '/share/home/202321008879/project/sandprocess/track/particle_properties.py'  # 替换为你的 npz 文件路径
data = np.load(npz_file)

# 查看文件中包含的所有 keys
print("Keys:", data.files)

# 逐个打印每个 key 对应的数据
for key in data.files:
    print(f"{key} :")
    print(data[key])