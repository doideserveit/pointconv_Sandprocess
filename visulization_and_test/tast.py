from skimage import measure
from skimage import io
from mayavi import mlab
import numpy as np

# 读取3D二值图像 (假设你的数据是3D体素)
label_data = io.imread('I:/20240623-original-p560test3.label.tif')
# 计算区域属性，返回所有连通区块的属性列表,每个属性都是一部字典，这个命令需要输入已经标记好的图像
# 即每个颗粒都有自己的标签值，而不是二值图（0，1）
label_data = np.transpose(label_data, (2, 1, 0))
regions = measure.regionprops(label_data)
print(f"区域的个数是：{len(regions)}")

# 用下述方法查看每个区域的信息（每个region是对应区域的字典）
# for region in regions:
#     print("Label:", region['Label'])
#     print("Area:", region['Area'])
#     print("Centroid:", region['Centroid'])

# 输出第一个区域的质心和体素数
# print(f"标签：{regions[13709].label}")
# print(f"质心坐标：{regions[13709].centroid}")
# print(f"体素数量：{regions[13709].area}")
# #
# # 计算体积，体素大小为 33.321 μm
# voxel_size = 0.033321
# volume = regions[13709].area * voxel_size**3  # 体积 = 体素数 * 体素大小的立方
# print(f"体积（mm³）：{volume}")


# 绘制指定标签的颗粒
# # 筛选标签值为 100 的区域，并扩展边界
# single_label = (label_data == 1800).astype(np.float32)
# padded_label = np.pad(single_label, pad_width=1, mode='constant', constant_values=0)
#
# # 绘制等值面，调整阈值
# mlab.contour3d(padded_label, contours=[0.5], opacity=1.0, color=(0.1, 0.5, 0.8))
#
# # 添加辅助元素
# mlab.title("Single Particle with Closed Boundary")
# mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
# mlab.show()
# 体渲染
# src = mlab.pipeline.scalar_field(label_data)
# mlab.pipeline.volume(src)
#
# # 添加辅助元素
# mlab.colorbar(title="Label Values")
# mlab.title("Volume Rendering of All Labels")
# mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
#
# mlab.show()


# print("数据形状：", label_data.shape)
# print("数据类型：", label_data.dtype)
#
#
# # 显示中间切片
# mid_slice = label_data.shape[0] // 2
# plt.imshow(label_data[mid_slice], cmap="gray")
# plt.title("Middle Slice of 3D Label Data")
# plt.show()
#
# # 除了第一个区域，还可以检查其他几个区域的质心、体素数和体积，以确保读取的所有区域信息都是合理的。
# for i, region in enumerate(regions[:5]):  # 检查前5个区域
#     # 当你对 enumerate(my_list) 进行迭代时，每次迭代都会从这个对象中获取一个元组，
#     # 元组的第一个元素是索引（默认从0开始），第二个元素是列表中的相应元素。
#     centroid = region.centroid
#     area = region.area
#     volume = area * voxel_size**3
#     print(f"区域 {i+1}: 质心坐标：{centroid}, 体素数量：{area}, 体积（μm³）：{volume}")
#
# # 可以统计 label_data 中的唯一值，看看是否得到了预期数量的分割标签。通常每个颗粒会有一个唯一标签值。
#
# unique_labels = np.unique(label_data)
# print("唯一标签数量：", len(unique_labels))
# print("唯一标签值：", unique_labels)
#
# # regionprops_table 可以生成一个数据表，方便进一步查看和验证属性。可以提取多个属性，例如质心、面积、边界框等。
#
# props = measure.regionprops_table(label_data, properties=
#     ['label', 'centroid', 'area', 'bbox'])
# df = pd.DataFrame(props)
# print(df.head())  # 查看前几行数据
