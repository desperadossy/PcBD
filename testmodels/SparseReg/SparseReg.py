import torch
import torch.nn.functional as F
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

def proximal_l1(v, lmbd):
    return torch.sign(v) * torch.clamp(v.abs() - lmbd, min=0.0)

def sparse_denoise_step(x, idx_knn, normals, sigma_h, sigma_d, lmbd=0.05, step_size=0.1, tau_clip=0.05):
    t0 = time.time()
    N, _ = x.shape
    device = x.device

    x_neighbors = x[idx_knn]             # (N, k, 3)
    diff = x_neighbors - x.unsqueeze(1)  # (N, k, 3)

    hi = torch.einsum("nkj,nj->nk", diff, normals)  # (N, k)
    dists = torch.norm(diff, dim=2)                # (N, k)

    theta = torch.exp(-dists**2 / (sigma_d**2))
    psi = torch.exp(-hi**2 / (sigma_h**2))
    eta = theta * psi / (hi.abs() + 1e-3)

    hi_centered = hi - hi.mean(dim=1, keepdim=True)
    grad_f = torch.einsum("nk,nkj->nj", hi_centered * eta, diff)  # (N, 3)

    n_j = normals[idx_knn]  # (N, k, 3)
    n_i = normals.unsqueeze(1)  # (N, 1, 3)
    reg = proximal_l1(n_i - n_j, lmbd).sum(dim=1)  # (N, 3)

    normals_new = normals - step_size * (grad_f + lmbd * reg)
    normals_new = F.normalize(normals_new, dim=1)

    hi_new = torch.einsum("nkj,nj->nk", diff, normals_new)
    psi_new = torch.exp(-hi_new**2 / (sigma_h**2))
    eta_new = theta * psi_new / (hi_new.abs() + 1e-3)

    tau = (eta_new * hi_new).sum(dim=1) / (eta_new.sum(dim=1) + 1e-8)
    tau = tau.clamp(min=-tau_clip, max=tau_clip)  # 防止过大跳跃
    x_new = x - tau.unsqueeze(1) * normals_new
    return x_new, normals_new

def SparseReg(x_input, k=25, sigma_h=0.008, sigma_d=0.02, lmbd=0.05, step_size=0.1, tau_clip=0.05, n_iter=20, early_stop=1e-4):
    t_start = time.time()
    x = x_input.squeeze(0).contiguous()
    device = x.device
    N = x.shape[0]

    idx_knn = knn_faiss(x, k=k)
    normals = F.normalize(torch.randn(N, 3, device=device), dim=1)

    for it in range(n_iter):
        x_new, normals = sparse_denoise_step(
            x, idx_knn, normals, sigma_h, sigma_d, lmbd, step_size, tau_clip
        )
        delta = (x_new - x).norm(dim=1).mean()
        if delta < early_stop:
            break
        x = x_new

    total_time = time.time() - t_start
    return x.unsqueeze(0)
