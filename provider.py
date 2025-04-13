import numpy as np
import torch
import torch.nn as nn
import sys
import importlib
import argparse
from torch.profiler import profile, record_function, ProfilerActivity
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, grouping_operation
from sklearn.decomposition import PCA
from torch_cluster import fps
from pytorch3d.ops import knn_points

def compute_roatation_matrix(sour_vec, dest_vec, sour_vertical_vec=None):
    # http://immersivemath.com/forum/question/rotation-matrix-from-one-vector-to-another/
    if np.linalg.norm(np.cross(sour_vec, dest_vec), 2) == 0 or np.abs(np.dot(sour_vec, dest_vec)) >= 1.0:
        if np.dot(sour_vec, dest_vec) < 0:
            return rotation_matrix(sour_vertical_vec, np.pi)
        return np.identity(3)
    alpha = np.arccos(np.dot(sour_vec, dest_vec))
    a = np.cross(sour_vec, dest_vec) / np.linalg.norm(np.cross(sour_vec, dest_vec), 2)
    c = np.cos(alpha)
    s = np.sin(alpha)
    R1 = [a[0] * a[0] * (1.0 - c) + c,
          a[0] * a[1] * (1.0 - c) - s * a[2],
          a[0] * a[2] * (1.0 - c) + s * a[1]]

    R2 = [a[0] * a[1] * (1.0 - c) + s * a[2],
          a[1] * a[1] * (1.0 - c) + c,
          a[1] * a[2] * (1.0 - c) - s * a[0]]

    R3 = [a[0] * a[2] * (1.0 - c) - s * a[1],
          a[1] * a[2] * (1.0 - c) + s * a[0],
          a[2] * a[2] * (1.0 - c) + c]

    R = np.matrix([R1, R2, R3])

    return R

def get_principle_dirs(pts):

    pts_pca = PCA(n_components=3)
    pts_pca.fit(pts)
    principle_dirs = pts_pca.components_
    principle_dirs /= np.linalg.norm(principle_dirs, 2, axis=0)

    return principle_dirs

def pca_alignment(pts, random_flag=False):

    pca_dirs = get_principle_dirs(pts)

    if random_flag:

        pca_dirs *= np.random.choice([-1, 1], 1)

    rotate_1 = compute_roatation_matrix(pca_dirs[2], [0, 0, 1], pca_dirs[1])
    pca_dirs = np.array(rotate_1 * pca_dirs.T).T
    rotate_2 = compute_roatation_matrix(pca_dirs[1], [1, 0, 0], pca_dirs[2])
    pts = np.array(rotate_2 * rotate_1 * np.matrix(pts.T)).T

    inv_rotation = np.array(np.linalg.inv(rotate_2 * rotate_1))

    return pts, inv_rotation

def pc_normalize(pc):

    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / (2*m)

    return pc
    
def l2_norm(vec, dim=-1):
    """Normalize the input vector"""
    norm = torch.norm(vec, p=2, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=1e-10)
    output = vec / norm
    return output


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1).contiguous())  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample+pad]
    return idx.int()


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=False, idx=None, use_dist=False, self_include=True):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous() # (B, N, 3)
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint)) # (B, 3, npoint)
    if idx is None:
        idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous(), include_self=self_include)
    grouped_xyz = grouping_operation(xyz, idx) # (B, 3, npoint, nsample)
    #grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)
    if use_dist:
        dist = torch.sqrt(torch.sum(grouped_xyz** 2, 1) + 1e-10)
        dist = (dist - dist.min(dim=2, keepdim=True)[0]) / (dist.max(dim=2, keepdim=True)[0] - dist.min(dim=2, keepdim=True)[0] + 1e-10)

    if points is not None:
        grouped_points = grouping_operation(points, idx) # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    if use_dist:
        return new_xyz, new_points, idx, grouped_xyz, dist
    else:
        return new_xyz, new_points, idx, grouped_xyz
        
def memory_hook(module, inputs, outputs):
    if not inputs:
        print(f"[{module.__class__.__name__}] WARNING: Empty inputs received in forward hook.")
        return 
    device = inputs[0].device
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    print(f"[{module.__class__.__name__}] allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")


def register_hooks(module):
    if isinstance(module, nn.ModuleList):  # 如果是 ModuleList，继续向下遍历
        for sub_module in module:
            register_hooks(sub_module)  # 递归注册
    else:
        module.register_forward_hook(memory_hook)  # 直接注册 hook


def group_random_sample(xyz, labels, n_points):
    choice = np.random.permutation(xyz.shape[0])
    xyz = xyz[choice[:n_points]]
    labels = labels[choice[:n_points]]
    if xyz.shape[0] < n_points:
        pzeros = np.zeros((n_points - xyz.shape[0], 3))
        lzeros = np.zeros((n_points - xyz.shape[0], 2))
        xyz = np.concatenate([xyz, pzeros])
        labels = np.concatenate((labels, lzeros))
    return xyz, labels

def random_sample(xyz, n_points):
    choice = np.random.permutation(xyz.shape[0])
    xyz = xyz[choice[:n_points]]
    if xyz.shape[0] < n_points:
        pzeros = np.zeros((n_points - xyz.shape[0], 3))
        xyz = np.concatenate([xyz, pzeros])
    return xyz

def upsample(xyz, n_points):
    curr = xyz.shape[0]
    need = n_points - curr
    if need < 0:
        return xyz[np.random.permutation(n_points)]
    while curr <= need:
        xyz = np.tile(xyz, (2, 1))
        need -= curr
        curr *= 2
    choice = np.random.permutation(need)
    xyz = np.concatenate((xyz, xyz[choice]))
    return xyz

def score_extract(points, scores, threshold=0.1, reserve_high=False):
    B, _, N = points.size()
    if reserve_high:
        high_score_mask = scores > threshold
    else:
        high_score_mask = scores < threshold
    points = points * high_score_mask
    nonzero_index = torch.max(high_score_mask, -1, keepdim=True)[1]
    replace_index = gather_operation(points, nonzero_index.int())  # Shape: (B, 1)
    points = torch.where(points == 0, replace_index, points)  # Broadcasting replace_index
    return points

def farthest_point_sampling(pcls, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3).
        num_pnts:  Target number of points.
    """
    ratio = 0.01 + num_pnts / pcls.size(1)
    sampled = []
    indices = []
    for i in range(pcls.size(0)):
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts]
        sampled.append(pcls[i:i+1, idx, :])
        indices.append(idx)
    sampled = torch.cat(sampled, dim=0)
    return sampled, indices

def light_patch_denoise(network, pcl_noisy, patch_size=1000, seed_k=3):
    """
    pcl_noisy:  Input point cloud, [N, 3]
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be [N, 3].'
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # [1, N, 3]
    num_patches = int(seed_k * N / patch_size)
    seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)  # [1, K, 3]
    patch_dists, point_idxs_in_main_pcd, patches = knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)

    patches = patches.view(-1, patch_size, 3)  # (K, M, 3)
    seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
    patches = patches - seed_pnts_1

    # Patch stitching preliminaries
    patch_dists, point_idxs_in_main_pcd = patch_dists[0], point_idxs_in_main_pcd[0]  # (K, M) (K, M)
    patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(1, patch_size)

    all_dists = torch.ones(num_patches, N) / 0
    all_dists = all_dists.cuda()
    all_dists = list(all_dists)
    patch_dists, point_idxs_in_main_pcd = list(patch_dists), list(point_idxs_in_main_pcd)

    for all_dist, patch_id, patch_dist in zip(all_dists, point_idxs_in_main_pcd, patch_dists):
        all_dist[patch_id] = patch_dist

    all_dists = torch.stack(all_dists, dim=0)  # (K, N)
    weights = torch.exp(-1 * all_dists)

    best_weights, best_weights_idx = torch.max(weights, dim=0)  # (N,) (N,) 最大权重、最大权重对应的块号
    patches_denoised = []
    i = 0
    patch_step = 1
    assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
    while i < num_patches:
        # print("Processed {:d}/{:d} patches.".format(i, num_patches))
        curr_patches = patches[i:i + patch_step]
        try:
            patches_denoised_temp = network(curr_patches)  # [K, M, 3]

        except Exception as e:
            print("=" * 100)
            print(e)
            print("=" * 100)
            print("If this is an Out Of Memory error, Seed_k_alpha might need to be increased to decrease patch_step.")
            print(
                "Additionally, if using multiple args.niters and a PyTorch3D ops, KNN, error arises, Seed_k might need to be increased to sample more patches for inference!")
            print("=" * 100)
            return
        patches_denoised.append(patches_denoised_temp)
        i += patch_step

    patches_denoised = torch.cat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_pnts_1
    pcl_denoised = []
    for pidx_in_main_pcd, patch in enumerate(best_weights_idx):
        mask = point_idxs_in_main_pcd[patch] == pidx_in_main_pcd
        if mask.sum() > 0:
            pcl_denoised.append(patches_denoised[patch][mask][0].unsqueeze(0))
        else:
            # 如果当前 patch 中没有包含该点，就保留原始点（兜底）
            pcl_denoised.append(pcl_noisy[0, pidx_in_main_pcd].unsqueeze(0))

    pcl_denoised = torch.cat(pcl_denoised, dim=0)  # [N, 3]
    return pcl_denoised

def patch_denoise(network, pcl_noisy, patch_size=1000, seed_k=3):
    """
    pcl_noisy:  Input point cloud, [N, 3]
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be [N, 3].'
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # [1, N, 3]
    num_patches = int(seed_k * N / patch_size)
    seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)  # [1, K, 3]
    patch_dists, point_idxs_in_main_pcd, patches = knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    
    patches = patches.view(-1, patch_size, 3)  # (K, M, 3)
    seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
    patches = patches - seed_pnts_1
    # Patch stitching preliminaries
    patch_dists, point_idxs_in_main_pcd = patch_dists[0], point_idxs_in_main_pcd[0]  # (K, M) (K, M)
    patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(1, patch_size)

    all_dists = torch.ones(num_patches, N) / 0
    all_dists = all_dists.cuda()
    all_dists = list(all_dists)
    patch_dists, point_idxs_in_main_pcd = list(patch_dists), list(point_idxs_in_main_pcd)

    for all_dist, patch_id, patch_dist in zip(all_dists, point_idxs_in_main_pcd, patch_dists):
        all_dist[patch_id] = patch_dist

    all_dists = torch.stack(all_dists, dim=0)  # (K, N)
    weights = torch.exp(-1 * all_dists)

    best_weights, best_weights_idx = torch.max(weights, dim=0)  # (N,) (N,) 最大权重、最大权重对应的块号
    patches_denoised = []
    i = 0
    patch_step = 1
    assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
    while i < num_patches:
        # print("Processed {:d}/{:d} patches.".format(i, num_patches))
        curr_patches = patches[i:i + patch_step]
        try:
            patches_denoised_temp = network[0](curr_patches)  # [K, M, 3]
            patches_denoised_temp = network[1](patches_denoised_temp)  # [K, M, 3]
            patches_denoised_temp = network[2](patches_denoised_temp)  # [K, M, 3]

        except Exception as e:
            print("=" * 100)
            print(e)
            print("=" * 100)
            print("If this is an Out Of Memory error, Seed_k_alpha might need to be increased to decrease patch_step.")
            print(
                "Additionally, if using multiple args.niters and a PyTorch3D ops, KNN, error arises, Seed_k might need to be increased to sample more patches for inference!")
            print("=" * 100)
            return
        patches_denoised.append(patches_denoised_temp)
        i += patch_step

    patches_denoised = torch.cat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_pnts_1
    pcl_denoised = []
    for pidx_in_main_pcd, patch in enumerate(best_weights_idx):
        mask = point_idxs_in_main_pcd[patch] == pidx_in_main_pcd
        if mask.sum() > 0:
            pcl_denoised.append(patches_denoised[patch][mask][0].unsqueeze(0))
        else:
            # 如果当前 patch 中没有包含该点，就保留原始点（兜底）
            pcl_denoised.append(pcl_noisy[0, pidx_in_main_pcd].unsqueeze(0))

    pcl_denoised = torch.cat(pcl_denoised, dim=0)  # [N, 3]
    return pcl_denoised

import math
def fibonacci_sphere_samples(num: int = 3, device='cpu'):
    """
    在球面上均匀采样方向向量（单位向量）
    Returns: List[Tensor (3,)]
    """
    points = []
    offset = 2.0 / num
    increment = math.pi * (3.0 - math.sqrt(5.0))  # 黄金角

    for i in range(num):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - y * y)
        phi = i * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        v = torch.tensor([x, y, z], dtype=torch.float32, device=device)
        points.append(v / torch.norm(v))  # 单位向量
    return points

def get_rotation_to_align_z(v: torch.Tensor) -> torch.Tensor:
    """
    返回旋转矩阵 R，使得 R @ P 将方向 v 旋转到 [0, 0, 1]
    """
    v = v / torch.norm(v)
    z = torch.tensor([0., 0., 1.], device=v.device)
    
    if torch.allclose(v, z):
        return torch.eye(3, device=v.device)
    if torch.allclose(v, -z):
        # 180 度旋转，绕任意垂直方向
        return torch.tensor([
            [-1.,  0.,  0.],
            [ 0., -1.,  0.],
            [ 0.,  0.,  1.]
        ], device=v.device)

    axis = torch.cross(v, z, dim=0)
    sin_angle = torch.norm(axis)
    axis = axis / sin_angle
    cos_angle = torch.dot(v, z)

    # 罗德里格斯公式构建旋转矩阵
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], device=v.device)

    R = torch.eye(3, device=v.device) + K + K @ K * ((1 - cos_angle) / (sin_angle ** 2))
    
    # ⚠️ 确保正交（防止数值误差）
    u, _, v_t = torch.linalg.svd(R)
    R_ortho = u @ v_t
    return R_ortho