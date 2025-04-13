import torch
import numpy as np
from scipy.spatial import Delaunay, cKDTree

def alpha_shape_edge_based(points_np, alpha):
    def add_edge(edges, i, j):
        if (i, j) in edges or (j, i) in edges:
            if (j, i) in edges:
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points_np)
    edges = set()
    edge_set = set()
    for simplex in tri.simplices:
        for a, b in [(0,1), (1,2), (2,0)]:
            i, j = simplex[a], simplex[b]
            edge = tuple(sorted((i, j)))
            edge_set.add(edge)
    
    N = points_np.shape[0]
    all_idx = np.arange(N)

    for i, j in edge_set:
        pi = points_np[i]
        pj = points_np[j]
        midpoint = (pi + pj) / 2
        direction = pj - pi
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-8:
            continue  # skip degenerate
        norm_dir = np.array([-direction[1], direction[0]]) / direction_norm

        r = alpha
        center1 = midpoint + norm_dir * (r / 2)
        center2 = midpoint - norm_dir * (r / 2)

        dists1 = np.linalg.norm(points_np - center1, axis=1)
        dists2 = np.linalg.norm(points_np - center2, axis=1)

        mask = (all_idx != i) & (all_idx != j)
        if (dists1[mask] >= r).all() or (dists2[mask] >= r).all():
            add_edge(edges, i, j)

    return edges

def AlphaShapes(points, alpha=0.015):
    points = points.squeeze()
    points_np = points[:, :2].contiguous().cpu().numpy()
    edges = alpha_shape_edge_based(points_np, alpha)

    boundary_indices = set()
    for i, j in edges:
        boundary_indices.add(i)
        boundary_indices.add(j)
    boundary_indices = list(boundary_indices)

    if not boundary_indices:
        raise ValueError("Please adjust alpha handly!")

    first_boundary_point = points[boundary_indices[0]]
    processed_points = torch.stack([
        points[i] if i in boundary_indices else first_boundary_point
        for i in range(len(points))
    ])
    return processed_points.unsqueeze(0)


"""
def alpha_shape(points, alpha):
    def add_edge(edges, i, j):
        if (i, j) in edges or (j, i) in edges:
            if (j, i) in edges:
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    for ia, ib, ic in tri.simplices:
        a = np.linalg.norm(points[ia] - points[ib])
        b = np.linalg.norm(points[ib] - points[ic])
        c = np.linalg.norm(points[ic] - points[ia])
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges
"""
