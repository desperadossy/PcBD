import torch

def compute_avg_nn_distance(points):
    B, N, _ = points.shape

    dists = torch.cdist(points, points, p=2)  

    dists[dists == 0] = torch.inf

    min_dists, _ = torch.min(dists, dim=-1)
    avg_nn_dist = torch.mean(min_dists, dim=-1)

    return avg_nn_dist

def ROR(points, radius=None, min_neighbors=None):

    B, N, _ = points.shape

    if radius is None:
        radius = 3.0 * compute_avg_nn_distance(points) 

    if min_neighbors is None:
        min_neighbors = max(5, int(0.001 * N))

    dists = torch.cdist(points, points, p=2)
    neighbor_counts = (dists < radius.unsqueeze(1).unsqueeze(2)).sum(dim=-1)  # (B, N)
    valid_mask = neighbor_counts >= min_neighbors  # (B, N)

    filtered_points = points.clone()
    filtered_points[~valid_mask.unsqueeze(-1).expand(-1, -1, 3)] = 0.0

    return filtered_points  # 形状仍然为 (B, N, 3)

