from skimage import measure
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mayavi import mlab

# 读取数据
label_data = io.imread('I:\sandprocess\data\origintest/origintest.label.tif')
# print(f"label数据的维数(z,y,x)：{label_data.shape}")
# 读取时，z作为数组的第0轴了，y为第1轴，因此要进行纠正，把2轴(avizo里为x轴的数据)的数据转到第0轴
corrected_data = np.transpose(label_data, (2, 1, 0))
# print(f"修改后数据的维数(x,y,z)：{corrected_data.shape}")
regions = measure.regionprops(corrected_data)
print(f"区域的个数是：{len(regions)}")
print(f"图片的维数：{corrected_data.ndim},数组维度(x,y,z)：{corrected_data.shape}")

# 区域1的信息
print(f"标签：{regions[0].label}")
print(f"质心坐标：{regions[0].centroid}")
print(f"体素数量：{regions[0].area}")

voxel_size = 0.033321
volume = regions[0].area * voxel_size**3  # 体积 = 体素数 * 体素大小的立方
print(f"体积（mm³）：{volume}")

# 使用mayavi绘制 3D 标签数据
# 单独绘制某标签的颗粒，contour的值是等值面，即把所有这个标签值的画出来
single_label = (label_data == 1).astype(np.float32) # single_label 是布尔掩码数据，
# 值为 0.0 表示标签不为 100，值为 1.0 表示标签为 100先将标签值100转化为1.0浮点数，其他转化为0.0
mlab.contour3d(single_label, contours=[0.5], opacity=1)# contour如果设置为一个整数（如 10），则生成 10 个等值面。
        # 如果设置为一个列表（如 [0.5, 1.0, 2.0]），则绘制指定值的等值面。
mlab.colorbar(title="Label Values")
mlab.title("3D Visualization of Label Image")
mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
mlab.show()

# # 体渲染
# src = mlab.pipeline.scalar_field(corrected_data)
# mlab.pipeline.volume(src)
#
# # 添加辅助元素
# mlab.colorbar(title="Label Values")
# mlab.title("Volume Rendering of All Labels")
# mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
#
# mlab.show()

# # 用matplotlib可视化体数据的表面，但是感觉跟avizo里的不像
# # 提取等值面
# verts, faces, normals, values = measure.marching_cubes(label_data, level=0)
#
# # 绘制等值面
# fig = plt.figure(figsize=(10, 10), dpi=300)
# ax = fig.add_subplot(111, projection='3d')
#
# # 使用 trisurf 绘制
# ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
#                 cmap='Spectral', lw=1)
#
# ax.set_title("3D Visualization of Label Image")
# plt.show()






