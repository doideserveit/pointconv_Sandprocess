# 示例字符串
input_str = "917,3;674,4;760,6;775,9;799,10;723,5;776,11;941,14;635,2;" \
"972,17;976,18;811,15;995,20;928,16;751,7;840,12;754,8;796,19;866,13;566,1;"

# 第一步：按照分号分割成多个部分
parts = input_str.split(';')

# 用于存储调换顺序后的结果
result = []

# 遍历每个分割出来的部分
for part in parts:
    # 按照逗号对每个部分再进行分割
    elements = part.split(',')
    # 确保这个部分正好有两个元素
    if len(elements) == 2:
        # 调换顺序
        swapped = f"{elements[1]},{elements[0]}"
        result.append(swapped)
    else:
        # 如果不是两个元素，可以选择忽略或者其他处理方式
        result.append(part)

print(result)