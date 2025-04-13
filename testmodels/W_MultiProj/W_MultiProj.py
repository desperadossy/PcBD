import torch
import torch.nn.functional as F

def construct_knn_graph(xyz, k=30):
    # xyz: (N, 3)
    dist = torch.cdist(xyz, xyz, p=2)  # (N, N)
    knn_idx = dist.topk(k=k+1, largest=False).indices[:, 1:]  # (N, k), skip self
    return knn_idx

def estimate_sigma(xyz, knn_idx):
    # Estimate sigma as average distance to k neighbors
    neighbors = xyz[knn_idx]  # (N, k, 3)
    center = xyz.unsqueeze(1)  # (N, 1, 3)
    dists = torch.norm(neighbors - center, dim=-1)  # (N, k)
    sigma = dists.mean().item()
    return sigma

def compute_weights(xyz, knn_idx, sigma):
    # xyz: (N, 3), knn_idx: (N, k)
    N, k = knn_idx.shape
    neighbors = xyz[knn_idx]  # (N, k, 3)
    center = xyz.unsqueeze(1)  # (N, 1, 3)
    dist2 = ((neighbors - center)**2).sum(dim=-1)  # (N, k)
    weights = torch.exp(-dist2 / (2 * sigma ** 2))  # (N, k)
    weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize
    return weights

def estimate_tangent_planes(xyz, knn_idx, weights):
    # xyz: (N, 3), knn_idx: (N, k), weights: (N, k)
    neighbors = xyz[knn_idx]  # (N, k, 3)
    mean = torch.sum(neighbors * weights.unsqueeze(-1), dim=1)  # (N, 3)
    diffs = neighbors - mean.unsqueeze(1)  # (N, k, 3)
    weighted_cov = torch.einsum('nki,nkj->nij', diffs * weights.unsqueeze(-1), diffs)  # (N, 3, 3)
    # Smallest eigenvector is normal
    eigvals, eigvecs = torch.linalg.eigh(weighted_cov)  # (N, 3), (N, 3, 3)
    normals = eigvecs[:, :, 0]  # (N, 3), smallest eigval
    intercepts = torch.sum(normals * mean, dim=1, keepdim=True)  # (N, 1)
    return normals, intercepts

def project_point_to_plane(xyz, normals, intercepts):
    # xyz: (N, 3), normals: (N, 3), intercepts: (N, 1)
    dot = (xyz * normals).sum(dim=1, keepdim=True)  # (N, 1)
    proj = xyz - (dot - intercepts) * normals  # (N, 3)
    return proj

def weighted_multi_projection(xyz, knn_idx, normals, intercepts, weights):
    # xyz: (N, 3), knn_idx: (N, k)
    N, k = knn_idx.shape
    projections = []
    for j in range(k):
        nj = normals[knn_idx[:, j]]  # (N, 3)
        cj = intercepts[knn_idx[:, j]]  # (N, 1)
        pij = xyz  # (N, 3)
        dot = (pij * nj).sum(dim=1, keepdim=True)
        proj = pij - (dot - cj) * nj  # (N, 3)
        projections.append(proj)
    projections = torch.stack(projections, dim=1)  # (N, k, 3)
    new_xyz = torch.sum(weights.unsqueeze(-1) * projections, dim=1)  # (N, 3)
    new_xyz = torch.nan_to_num(new_xyz, nan=0.0)  # Replace NaNs in result
    return new_xyz

def W_MultiProj(xyz, k=30, sigma=None):
    # xyz: (1, N, 3)
    xyz = xyz.squeeze(0)  # (N, 3)
    knn_idx = construct_knn_graph(xyz, k)
    if sigma is None:
        sigma = estimate_sigma(xyz, knn_idx)
    weights = compute_weights(xyz, knn_idx, sigma)
    normals, intercepts = estimate_tangent_planes(xyz, knn_idx, weights)
    denoised = weighted_multi_projection(xyz, knn_idx, normals, intercepts, weights)
    return denoised.unsqueeze(0)  # (1, N, 3)
