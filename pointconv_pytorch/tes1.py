import torch

# 模拟一个预测结果，形状为 (3, 5)，表示 3 个样本，每个样本有 5 个类别预测值
pred = torch.tensor([
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.2, 0.3, 0.5, 0.1, 0.4]
])

# 获取每个样本预测的类别索引
pred_choice = pred.max(1)[1]

print("预测结果张量:")
print(pred)
print("pred.max:")
print(pred.max)
print("pred.max(1):")
print(pred.max(1))
print("每个样本预测的类别索引:")
print(pred_choice)