import torch
import numpy as np
import faiss
import time


def knn_faiss(x, k):
    t0 = time.time()
    x_np = x.cpu().numpy().astype(np.float32)
    index = faiss.IndexFlatL2(3)
    index.add(x_np)
    _, idx = index.search(x_np, k)
    return torch.from_numpy(idx).to(x.device)

def compute_lcs_batch(p_neighbors, centers):
    t0 = time.time()
    centered = p_neighbors - centers[:, None, :]
    cov = torch.matmul(centered.transpose(1, 2), centered) / (centered.size(1) + 1e-8)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = torch.argsort(eigvals, descending=True, dim=1)
    eigvecs_sorted = torch.gather(eigvecs, 2, idx[:, None, :].expand(-1, 3, -1))
    ci = eigvecs_sorted[:, :, 0]
    di = eigvecs_sorted[:, :, 1]
    return ci, di

def pca_normal_batched(local_points, mask):
    device = local_points.device
    N, M, _ = local_points.shape

    # (N, M, 1)
    mask = mask.unsqueeze(-1)
    masked_points = local_points * mask
    count = mask.sum(dim=1, keepdim=True).float()  # (N, 1, 1)
    # 至少要 3 个点，否则结果无效
    valid_mask = (count.view(-1) >= 3)

    # 防止分母为 0
    count = count.clamp(min=1e-6)

    mean = masked_points.sum(dim=1, keepdim=True) / count  # (N, 1, 3)
    centered = (local_points - mean) * mask  # (N, M, 3)

    cov = torch.matmul(centered.transpose(1, 2), centered) / count  # (N, 3, 3)
    eigvals, eigvecs = torch.linalg.eigh(cov)                       # (N, 3), (N, 3, 3)

    # 找到最小特征值对应的向量
    idx = torch.argsort(eigvals, dim=1)       # (N, 3)
    idx_min = idx[:, 0].unsqueeze(1).unsqueeze(2)    # (N, 1, 1)
    idx_min_expanded = idx_min.expand(-1, 3, 1)      # (N, 3, 1)
    normals = torch.gather(eigvecs, 2, idx_min_expanded).squeeze(-1)  # (N, 3)

    # 对无效邻域做 fallback 处理
    fallback_mask = (~valid_mask)
    if fallback_mask.any():
        normals[fallback_mask] = 0.0

    return normals

def estimate_in_direction_vectorized(x, idx_knn, dirs, h_list, sigma=0.01, dist_filter_factor=2.0, M_max=50):
    t0 = time.time()
    device = x.device
    N, k = idx_knn.shape

    x_neighbors = x[idx_knn]  # (N, k, 3)
    x_center = x.unsqueeze(1)  # (N, 1, 3)
    diffs = x_neighbors - x_center  # (N, k, 3)
    dists = torch.norm(diffs, dim=2)  # (N, k)
    mean_dist = dists.mean(dim=1, keepdim=True)
    mask_dist = dists < dist_filter_factor * mean_dist  # (N, k)

    best_errors = torch.full((N,), float("inf"), device=device)
    best_p = x.clone()
    best_n = torch.zeros((N, 3), device=device)

    for h in h_list:
        half_h = h / 2.0
        proj = torch.einsum('nkj,nj->nk', diffs, dirs)
        mask_h = torch.abs(proj) < half_h
        mask = mask_dist & mask_h  # (N, k)

        sort_scores = mask.to(torch.float32)  # 更安全的排序，不涉及 inf
        topk_idx = sort_scores.argsort(dim=1, descending=True)[:, :M_max]  # (N, M_max)

        batch_indices = torch.arange(N, device=device).unsqueeze(1).expand(-1, M_max)
        local_pts = x_neighbors[batch_indices, topk_idx]  # (N, M_max, 3)
        mask_gather = mask[batch_indices, topk_idx]  # (N, M_max)

        normals = pca_normal_batched(local_pts, mask_gather)  # (N, 3)
        mean = (local_pts * mask_gather.unsqueeze(-1)).sum(dim=1) / mask_gather.sum(dim=1, keepdim=True)
        proj_dists = torch.einsum('nij,nj->ni', local_pts - mean.unsqueeze(1), normals)
        error = (proj_dists ** 2 * mask_gather).sum(dim=1) / mask_gather.sum(dim=1)  # (N,)

        update_mask = error < best_errors
        best_errors = torch.where(update_mask, error, best_errors)
        offset = torch.einsum('ni,ni->n', x - mean, normals)
        best_p = torch.where(update_mask.unsqueeze(-1), x - offset.unsqueeze(-1) * normals, best_p)
        best_n = torch.where(update_mask.unsqueeze(-1), normals, best_n)

    error_val = best_errors.clamp(min=1e-10)
    weights = torch.exp(-error_val / (2 * sigma ** 2))

    return best_p, best_n, weights




def estimate_all_directions(x_input, k=50, h_list=None, sigma=0.01, dist_filter_factor=2.0):
    t0 = time.time()
    x = x_input.squeeze(0)
    device = x.device
    N = x.shape[0]

    if h_list is None:
        tmp_idx = knn_faiss(x.contiguous(), k=10)
        dists_all = [(x[tmp_idx[i]] - x[i]).norm(dim=1).mean().item() for i in range(N)]
        mean_d = float(np.mean(dists_all))
        h_list = [mean_d, 2 * mean_d, 4 * mean_d]

    idx_knn = knn_faiss(x.contiguous(), k=k)
    neighbors = x[idx_knn]  # (N, k, 3)
    ci, di = compute_lcs_batch(neighbors, x)
    dirs = [ci, -ci, di, -di]

    est_all, normal_all, weight_all = [], [], []
    for d in dirs:
        est, nrm, w = estimate_in_direction_vectorized(
            x, idx_knn, d, h_list, sigma=sigma,
            dist_filter_factor=dist_filter_factor, M_max=k
        )
        est_all.append(est)
        normal_all.append(nrm)
        weight_all.append(w)

    return est_all, normal_all, weight_all

def aggregate_estimates_vectorized(x_estimates, normals, weights, device='cuda', reg=1e-6):
    t0 = time.time()
    N = x_estimates[0].shape[0]
    D = len(x_estimates)  # 通常为 4

    p = torch.stack(x_estimates, dim=0)  # (D, N, 3)
    n = torch.stack(normals, dim=0)      # (D, N, 3)
    w = torch.stack(weights, dim=0)      # (D, N)

    w = w.unsqueeze(-1).unsqueeze(-1)    # (D, N, 1, 1)
    nT = n.unsqueeze(-1)                 # (D, N, 3, 1)
    n_outer = nT @ nT.transpose(-2, -1)  # (D, N, 3, 3)
    P = torch.eye(3, device=device).reshape(1, 1, 3, 3) - n_outer  # (D, N, 3, 3)

    P_weighted = w * P  # (D, N, 3, 3)
    A = P_weighted.sum(dim=0) + reg * torch.eye(3, device=device).unsqueeze(0)  # (N, 3, 3)
    b = (P_weighted @ p.unsqueeze(-1)).sum(dim=0).squeeze(-1)  # (N, 3)

    try:
        q = torch.linalg.solve(A, b)  # (N, 3)
    except:
        q = b  # fallback

    return q.unsqueeze(0)

def AdaMLS(x_noisy, k=50, h_list=None, sigma=0.01, dist_filter_factor=1.0, reg=1e-6, max_iter=2):
    device = x_noisy.device
    x_denoised = x_noisy.clone()

    for iter_id in range(max_iter):
        t0 = time.time()
        ests, norms, ws = estimate_all_directions(
            x_denoised, k=k, h_list=h_list,
            sigma=sigma, dist_filter_factor=dist_filter_factor
        )
        x_next = aggregate_estimates_vectorized(ests, norms, ws, device=device, reg=reg)
        x_denoised = x_next

    return x_denoised
