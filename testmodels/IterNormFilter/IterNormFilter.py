import torch
from pointnet2_ops import pointnet2_utils

def batch_svd_safe(cov_matrix):
    U, S, Vh = torch.linalg.svd(cov_matrix)
    normals = Vh[:, :, -1, :]
    normals = torch.nn.functional.normalize(normals, dim=-1)
    normals = torch.nan_to_num(normals, nan=0.0)
    return normals

def estimate_normals_gpu_safe(points, neighbor_idx):
    B, N, K = neighbor_idx.shape
    grouped = pointnet2_utils.grouping_operation(points.transpose(1, 2).contiguous(), neighbor_idx)
    grouped = grouped.permute(0, 2, 3, 1)
    center = points.unsqueeze(2)
    neighbors_centered = grouped - center
    cov = neighbors_centered.transpose(-1, -2) @ neighbors_centered
    normals = batch_svd_safe(cov)
    return normals

def bilateral_filter_normals_gpu_safe(normals, points, neighbor_idx, avg_knn_dist, alpha=2.5, beta=0.5, gamma=1.0):
    B, N, K = neighbor_idx.shape
    grouped_pts = pointnet2_utils.grouping_operation(points.transpose(1, 2).contiguous(), neighbor_idx).permute(0, 2, 3, 1)
    grouped_nrm = pointnet2_utils.grouping_operation(normals.transpose(1, 2).contiguous(), neighbor_idx).permute(0, 2, 3, 1)
    center_pts = points.unsqueeze(2)
    center_nrm = normals.unsqueeze(2)

    dist = torch.norm(grouped_pts - center_pts, dim=-1)
    sigma_s = beta * avg_knn_dist.unsqueeze(-1)
    sigma_s = sigma_s.clamp(min=1e-6)
    w_s = torch.exp(- (dist ** 2) / (2 * sigma_s ** 2))

    dot = (grouped_nrm * center_nrm).sum(dim=-1)
    dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
    angle_diff = torch.acos(dot)
    sigma_r = gamma * (angle_diff.std(dim=-1, keepdim=True) + 1e-6)
    w_r = torch.exp(- (angle_diff ** 2) / (2 * sigma_r ** 2))

    weights = w_s * w_r
    weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-8
    filtered = (grouped_nrm * weights.unsqueeze(-1)).sum(dim=2) / weights_sum
    filtered = torch.nan_to_num(filtered, nan=0.0)
    filtered = torch.nn.functional.normalize(filtered, dim=-1)
    filtered = torch.nan_to_num(filtered, nan=0.0)
    return filtered

def update_points_gpu_safe(points, normals, avg_knn_dist, step_ratio=0.1):
    step = step_ratio * avg_knn_dist.unsqueeze(-1)
    return points + step * normals

def IterNormFilter(point_cloud, iterations=5, nsample=32, alpha=2.5, beta=0.5, gamma=1.0, step_ratio=0.03):
    assert point_cloud.dim() == 3
    B, N, _ = point_cloud.shape
    xyz = point_cloud
    with torch.no_grad():
        dists = torch.cdist(xyz, xyz)
        knn_dists, _ = torch.topk(dists, k=nsample + 1, largest=False)
        avg_knn_dist = knn_dists[:, :, 1:].mean(dim=-1)
        radius = alpha * avg_knn_dist
        max_radius = radius.max().item()
        neighbor_idx = pointnet2_utils.ball_query(max_radius, nsample, xyz.contiguous(), xyz.contiguous())
    pts = xyz
    for _ in range(iterations):
        normals = estimate_normals_gpu_safe(pts, neighbor_idx)
        normals = bilateral_filter_normals_gpu_safe(normals, pts, neighbor_idx, avg_knn_dist,
                                                    alpha=alpha, beta=beta, gamma=gamma)
        pts = update_points_gpu_safe(pts, normals, avg_knn_dist, step_ratio=step_ratio)
    return pts
