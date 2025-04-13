import torch

def adaptive_k(N, beta=1.5):
    return max(10, int(beta * (N ** (1/3))))

def adaptive_std_multiplier(N):
    gamma = 1.0
    delta = 0.2
    return max(0.5, gamma + delta * (torch.log10(torch.tensor(N)) - 3))

def SOR(points, k=None, std_multiplier=None):
    
    B, N, _ = points.shape
    if k is None:
        k = adaptive_k(N) 
    if std_multiplier is None:
        std_multiplier = adaptive_std_multiplier(N)

    dists = torch.cdist(points, points, p=2)  
    dists, indices = torch.sort(dists, dim=-1)  
    k_dists = dists[:, :, 1:k+1] 
    
    # 计算 k 近邻均值距离
    mean_dists = k_dists.mean(dim=-1)  # (B, N)

    # 计算全局均值和标准差
    mean = mean_dists.mean(dim=-1, keepdim=True)  # (B, 1)
    std = mean_dists.std(dim=-1, keepdim=True)  # (B, 1)

    # 识别离群点
    threshold = mean + std_multiplier * std  # (B, 1)
    valid_mask = mean_dists <= threshold  # (B, N)，True 代表正常点，False 代表离群点

    # 复制点云
    filtered_points = points.clone()

    # 将离群点设为 (0,0,0)
    filtered_points[~valid_mask.unsqueeze(-1).expand(-1, -1, 3)] = 0.0

    return filtered_points  # 形状仍然为 (B, N, 3)