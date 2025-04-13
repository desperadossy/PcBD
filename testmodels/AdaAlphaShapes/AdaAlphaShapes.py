import torch
import numpy as np
from scipy.spatial import Delaunay
from pointnet2_ops.pointnet2_utils import ball_query
from collections import defaultdict
import faulthandler
faulthandler.enable()

def AdaAlphaShapes(points: torch.Tensor, radius: float = 0.3, nsample: int = 256):
    """
    Adaptive Alpha Shape using ball_query and edge-based psavg (accelerated with edge_dict).

    Args:
        points (torch.Tensor): (1, N, 3) point cloud tensor (on GPU)
        radius (float): Radius for ball_query neighborhood
        nsample (int): Max samples per ball neighborhood

    Returns:
        torch.Tensor: (1, N, 3) processed point cloud
    """
    assert points.ndim == 3 and points.shape[0] == 1
    device = points.device
    B, N, _ = points.shape
    points_2d = points[0, :, :2]
    points_2d_np = points_2d.detach().cpu().numpy()

    # Step 1: Delaunay triangulation (on CPU)
    tri = Delaunay(points_2d_np)
    simplices = torch.from_numpy(tri.simplices).to(device)

    # Step 2: Extract all edges
    edge_set = set()
    for simplex in tri.simplices:
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            edge = tuple(sorted((simplex[a], simplex[b])))
            edge_set.add(edge)
    edges = list(edge_set)
    edge_tensor = torch.tensor(edges, dtype=torch.long, device=device)  # (E, 2)

    # Step 3: Build edge_dict for fast access
    edge_dict = defaultdict(list)
    for idx, (j, k) in enumerate(edges):
        edge_dict[j].append((j, k))
        edge_dict[k].append((j, k))  # 双向加入

    # Step 4: ball_query + fast local Delaunay edge-based psavg
    idx_ball = ball_query(radius, nsample, points.contiguous(), points.contiguous())  # (1, N, nsample)
    psavg = torch.zeros(N, device=device)

    for i in range(N):
        neighbors = idx_ball[0, i].unique()
        neighbors = neighbors[neighbors != -1]
        neighbor_set = set(neighbors.tolist())

        local_edges = []
        seen = set()
        for n in neighbors.tolist():
            for j, k in edge_dict.get(n, []):
                edge = tuple(sorted((j, k)))
                if edge in seen:
                    continue
                seen.add(edge)
                if j in neighbor_set and k in neighbor_set:
                    pj = points_2d_np[j]
                    pk = points_2d_np[k]
                    local_edges.append(np.linalg.norm(pj - pk))

        if local_edges:
            psavg[i] = torch.tensor(np.mean(local_edges), device=device)
        else:
            psavg[i] = 0.0

    # Step 5: compute triangle circumradius and filter
    ia, ib, ic = simplices[:, 0], simplices[:, 1], simplices[:, 2]
    pa, pb, pc = points[0, ia, :2], points[0, ib, :2], points[0, ic, :2]
    a = torch.norm(pa - pb, dim=1)
    b = torch.norm(pb - pc, dim=1)
    c = torch.norm(pc - pa, dim=1)
    s = (a + b + c) / 2
    area = torch.sqrt(torch.clamp(s * (s - a) * (s - b) * (s - c), min=1e-12))
    circum_r = a * b * c / (4 * area)
    alpha = (psavg[ia] + psavg[ib] + psavg[ic]) / 3
    valid = (circum_r <= alpha)

    # Step 6: extract boundary edges
    valid_simplices = simplices[valid]
    edges_valid = torch.cat([
        valid_simplices[:, [0, 1]],
        valid_simplices[:, [1, 2]],
        valid_simplices[:, [2, 0]],
    ], dim=0)
    edges_sorted = torch.sort(edges_valid, dim=1)[0]
    edges_unique, counts = torch.unique(edges_sorted, return_counts=True, dim=0)
    boundary_edges = edges_unique[counts == 1]
    boundary_idx = torch.unique(boundary_edges)

    # Step 7: replace non-boundary points
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[boundary_idx] = True
    first_boundary_point = points[0, boundary_idx[0]]
    processed = points[0].clone()
    processed[~mask] = first_boundary_point

    return processed.unsqueeze(0)
