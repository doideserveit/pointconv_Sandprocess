import torch
from torch_cluster import fps
print(torch.__version__)          # PyTorch版本，如1.10.0
print(torch.version.cuda)          # 对应的CUDA版本，如11.3（若为CPU版本显示None）


def main():
    # 生成 1000 个随机的三维点
    num_points = 1000
    points = torch.rand((num_points, 3), dtype=torch.float32)
    
    # 如果有 cuda 则使用，否则使用 cpu
    print(f"cuda: {torch.cuda.is_available()}")
    device = 'cuda' 
    points = points.to(device)
    
    # 创建对应的 batch 信息，全部归为一个 batch
    batch = torch.zeros(points.size(0), dtype=torch.long, device=device)
    
    # 设置采样比例，例如采样 10% 的点
    ratio = 0.1
    sampled_indices = fps(points, batch, ratio=ratio)
    
    print("采样后的索引个数:", sampled_indices.shape[0])
    print("采样的索引:", sampled_indices.cpu().numpy())

if __name__ == '__main__':
    main()



