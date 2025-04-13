import torch

def compute_avg_nn_distance(points):
    """计算点云的平均最近邻距离 (Average Nearest Neighbor Distance)"""
    B, N, _ = points.shape
    k = int(N ** (1/3))
    dists = torch.cdist(points, points, p=2)  # 计算欧几里得距离 (B, N, N)
    dists[dists == 0] = torch.inf  # 过滤自身点
    min_dists, _ = torch.sort(dists, dim=-1)  # (B, N, N)
    k_dists = min_dists[:, :, 1:k+1]  # 取最近的 k 个邻居（去掉自身）
    avg_nn_dist = k_dists.mean(dim=-1).mean(dim=-1)  # 计算均值 (B,)
    return avg_nn_dist

def DBSCAN(points, eps=None, min_pts=None):

    B, N, _ = points.shape
    if eps==None:
        eps = 1.5 * compute_avg_nn_distance(points)
    if min_pts==None:
        min_pts = int(N ** (1/3))
    labels = torch.full((B, N), -1, dtype=torch.long, device=points.device)  # -1 代表离群点
    cluster_id = 0  # 聚类编号

    # 计算距离矩阵 (B, N, N)
    dists = torch.cdist(points, points, p=2)  # 计算欧几里得距离
    neighbor_mask = dists < eps  # (B, N, N)，True 表示在邻域内

    for b in range(B):
        visited = torch.zeros(N, dtype=torch.bool, device=points.device)  # 记录访问点
        cluster_id = 0

        for i in range(N):
            if visited[i] or labels[b, i] != -1:
                continue  # 如果已访问或已分类，则跳过

            neighbors = neighbor_mask[b, i].nonzero().squeeze(1)  # 获取邻域内的索引
            
            if len(neighbors) < min_pts:
                continue  # 不满足 min_pts，则继续，仍保持 -1（离群点）

            # 形成新聚类
            cluster_id += 1
            labels[b, i] = cluster_id
            visited[i] = True

            # 逐步扩展簇
            queue = neighbors.tolist()
            while queue:
                p = queue.pop(0)
                if visited[p]:
                    continue
                visited[p] = True
                labels[b, p] = cluster_id  # 赋予聚类ID

                # 继续扩展
                new_neighbors = neighbor_mask[b, p].nonzero().squeeze(1)
                if len(new_neighbors) >= min_pts:
                    queue.extend(new_neighbors.tolist())  # 只有核心点的邻域才继续扩展

    # 复制原始点云
    filtered_points = points.clone()

    # 将离群点设为 (0,0,0)，确保形状匹配
    filtered_points[(labels == -1).unsqueeze(-1).expand(-1, -1, 3)] = 0.0

    return filtered_points  # (B, N, 3)