import torch

def BilateralFilter(points, k=16, alpha=2.5, beta=0.5, gamma=1.0):
    """
    自适应双边滤波（Bilateral Filtering），支持 [1, N, 3] 输入。
    参数：
        points: Tensor, [1, N, 3]
        k: 用于估计局部密度的最近邻数量
        alpha: radius = alpha * avg_knn_dist
        beta: sigma_s = beta * avg_knn_dist
        gamma: sigma_r = gamma * 局部z方向标准差
    返回：
        filtered_points: Tensor, [1, N, 3]
    """
    assert points.dim() == 3 and points.shape[0] == 1, "Input must be [1, N, 3]"
    device = points.device
    N = points.shape[1]
    p = points[0]  # [N, 3]

    # Step 1: 计算 kNN 距离
    dists = torch.cdist(p.unsqueeze(0), p.unsqueeze(0))[0]  # [N, N]
    knn_dists, _ = torch.topk(dists, k=k+1, largest=False)  # 包含自己
    avg_knn_dist = knn_dists[:, 1:].mean(dim=1)  # [N]

    # 自适应参数
    radius = alpha * avg_knn_dist  # [N]
    sigma_s = beta * avg_knn_dist  # [N]

    # Step 2: 估算每点局部 z 方差，作为 sigma_r
    z_vals = p[:, 2]  # [N]
    sigma_r = torch.zeros(N, device=device)
    neighbors_idx = [(dists[i] < radius[i]).nonzero(as_tuple=True)[0] for i in range(N)]
    for i, idx in enumerate(neighbors_idx):
        if idx.numel() < 2:
            sigma_r[i] = 1e-6  # 或设为 avg_knn_dist[i] 的倍数等默认值
        else:
            sigma_r[i] = z_vals[idx].std() + 1e-6

    # Step 3: 双边滤波
    filtered = torch.zeros_like(p)
    for i in range(N):
        idx = neighbors_idx[i]
        neighbor_pts = p[idx]  # [M, 3]
        diff = neighbor_pts - p[i]  # [M, 3]
        dist_s = diff.norm(dim=1)
        dist_r = diff.norm(dim=1)

        w_s = torch.exp(- (dist_s ** 2) / (2 * sigma_s[i] ** 2))
        w_r = torch.exp(- (dist_r ** 2) / (2 * (gamma * sigma_r[i]) ** 2))
        weights = w_s * w_r
        weights_sum = weights.sum() + 1e-8

        filtered[i] = torch.sum(neighbor_pts * weights[:, None], dim=0) / weights_sum

    return filtered.unsqueeze(0)  # [1, N, 3]