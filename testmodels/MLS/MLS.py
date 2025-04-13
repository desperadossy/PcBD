import torch
import torch.nn.functional as F
import faiss

def safe_norm(v, eps=1e-14):
    return torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True) + eps)

def knn_indices_faiss(x, k=16):
    x = x.squeeze(0).contiguous()  # (N, 3)
    x_np = x.cpu().numpy().astype('float32')
    index = faiss.IndexFlatL2(3)
    index.add(x_np)
    _, I = index.search(x_np, k)
    return torch.from_numpy(I).to(x.device)

def estimate_planes_and_normals_batch(r, neighbors, h, eps=1e-9):
    # r: (N, 3), neighbors: (N, k, 3)
    diff = neighbors - r.unsqueeze(1)  # (N, k, 3)
    dist2 = (diff ** 2).sum(dim=2)     # (N, k)
    w = torch.exp(-dist2 / (h ** 2))   # (N, k)
    sum_w = w.sum(dim=1, keepdim=True)  # (N, 1)

    # 避免退化
    valid = (sum_w > 1e-14).float()
    w = w * valid

    # 加权协方差矩阵
    w_unsq = w.unsqueeze(2)  # (N, k, 1)
    diff_w = diff * w_unsq   # (N, k, 3)
    C = torch.matmul(diff_w.transpose(1, 2), diff)  # (N, 3, 3)
    eye = torch.eye(3, device=r.device).unsqueeze(0)
    C = C + eps * eye

    # 特征值分解
    eigvals, eigvecs = torch.linalg.eigh(C)  # (N, 3), (N, 3, 3)
    min_eigvec = eigvecs[:, :, 0]            # (N, 3)
    min_eigvec = min_eigvec / safe_norm(min_eigvec)  # 归一化
    D = torch.sum(min_eigvec * r, dim=1, keepdim=True)  # (N, 1)
    return min_eigvec, D

def mls_project_batch_fast(x_est, x_ref, knn_idx, h=0.05, poly_degree=1, eps=1e-9):
    N, k = knn_idx.shape
    r = x_est.squeeze(0)  # (N, 3)
    neighbors = x_ref[knn_idx]  # (N, k, 3)

    n, D = estimate_planes_and_normals_batch(r, neighbors, h, eps)

    dist_to_plane = torch.sum(n * r, dim=1, keepdim=True) - D  # (N, 1)
    q = r - dist_to_plane * n  # (N, 3)

    diff = neighbors - q.unsqueeze(1)  # (N, k, 3)
    dist2 = (diff ** 2).sum(dim=2)     # (N, k)
    w = torch.exp(-dist2 / (h ** 2))   # (N, k)

    tmp = torch.tensor([1., 0., 0.], device=r.device).expand_as(n)
    alt = torch.tensor([0., 1., 0.], device=r.device).expand_as(n)
    dot_tmp = torch.sum(tmp * n, dim=1, keepdim=True)
    tmp = torch.where(dot_tmp > 0.9, alt, tmp)
    proj = tmp - torch.sum(tmp * n, dim=1, keepdim=True) * n
    u0 = proj / safe_norm(proj)
    v0 = torch.cross(n, u0, dim=1)
    v0 = v0 / safe_norm(v0)

    xi = torch.sum(diff * u0.unsqueeze(1), dim=2)  # (N, k)
    yi = torch.sum(diff * v0.unsqueeze(1), dim=2)  # (N, k)
    fi = torch.sum(diff * n.unsqueeze(1), dim=2)   # (N, k)

    if poly_degree == 1:
        M = torch.stack([torch.ones_like(xi), xi, yi], dim=2)  # (N, k, 3)
    elif poly_degree == 2:
        M = torch.stack([
            torch.ones_like(xi), xi, yi,
            xi * xi, xi * yi, yi * yi
        ], dim=2)  # (N, k, 6)
    else:
        raise ValueError("Only degree 1 or 2 supported")

    Wsqrt = torch.sqrt(w + 1e-14).unsqueeze(2)  # (N, k, 1)
    M_ = M * Wsqrt
    f_ = fi * Wsqrt.squeeze(2)  # (N, k)

    A = torch.matmul(M_.transpose(1, 2), M_)  # (N, d, d)
    b = torch.matmul(M_.transpose(1, 2), f_.unsqueeze(2))  # (N, d, 1)
    A = A + eps * torch.eye(A.shape[-1], device=A.device).unsqueeze(0)

    a = torch.linalg.solve(A, b).squeeze(2)  # (N, d)
    g0 = a[:, 0:1]  # (N, 1)
    r_new = q + g0 * n  # (N, 3)
    return r_new.unsqueeze(0)

def MLS(x_init, k=16, h=0.05, poly_degree=1, num_iter=5, eps=1e-9):
    x = x_init.clone()
    for _ in range(num_iter):
        knn_idx = knn_indices_faiss(x, k=k)
        x_ref = x.squeeze(0)
        x = mls_project_batch_fast(x, x_ref, knn_idx, h=h, poly_degree=poly_degree, eps=eps)
    return x

if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 4096
    xyz_clean = torch.rand((1, N, 3), device=device) * 0.5
    noise = 0.01 * torch.randn_like(xyz_clean)
    xyz_noisy = xyz_clean + noise

    start = time.time()
    xyz_denoised = MLS_fast(
        xyz_noisy, k=16, h=0.05, poly_degree=2, num_iter=5, eps=1e-9
    )
    end = time.time()

    print("Done in {:.4f} seconds".format(end - start))
    print("Output shape:", xyz_denoised.shape)
    print("MSE vs clean:", torch.mean((xyz_denoised - xyz_clean) ** 2).item())
