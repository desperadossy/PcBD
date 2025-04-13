import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def NormalComparing(points: torch.Tensor,  k: int = 60, angle_threshold_deg: float = 90.0):
    """
    基于方向角变化提取边界点（论文图4/图5逻辑）

    Args:
        points: (1, N, 3) 点云数据
        theta: (3,) 投影方向向量
        k: 邻域点数
        angle_threshold_deg: 角度间隔判据（如60°）

    Returns:
        (1, N, 3) 点云，非边界点替换为第一个边界点
    """
    theta=torch.tensor([0, 0, 1.0], device=points.device)
    assert points.ndim == 3 and points.shape[0] == 1
    device = points.device
    N = points.shape[1]
    points_3d = points[0]  # (N, 3)

    # Step 1: 将点云投影到与 theta 垂直的 2D 平面
    theta = theta / torch.norm(theta)
    z_axis = theta
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
    if torch.allclose(x_axis, z_axis):
        x_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
    x_axis = x_axis - torch.dot(x_axis, z_axis) * z_axis
    x_axis = x_axis / torch.norm(x_axis)
    y_axis = torch.cross(z_axis, x_axis, dim=0)

    R = torch.stack([x_axis, y_axis, z_axis], dim=1)
    points_proj = (points_3d @ R)[:, :2]  # (N, 2)
    points_np = points_proj.cpu().numpy()

    # Step 2: 用 sklearn 进行 KNN 查询（邻域内角度分布）
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(points_np)
    _, indices = nbrs.kneighbors(points_np)
    indices = indices[:, 1:]  # 排除自身

    boundary_mask = torch.zeros(N, dtype=torch.bool, device=device)

    for i in range(N):
        pi = points_np[i]
        neighbors = points_np[indices[i]]
        vecs = neighbors - pi  # (k, 2)
        angles = np.arctan2(vecs[:, 1], vecs[:, 0]) * 180 / np.pi  # (-180, 180]
        angles = np.mod(angles, 360)  # [0, 360)
        angles_sorted = np.sort(angles)[::-1]  # 降序排列

        # δ序列计算
        delta = [360 - angles_sorted[0]]
        for j in range(1, k):
            delta.append(angles_sorted[j - 1] - angles_sorted[j])
        delta.append(angles_sorted[-1])
        delta = np.array(delta)

        if np.max(delta) > angle_threshold_deg:
            boundary_mask[i] = True

    # Step 3: 替换非边界点
    boundary_idx = torch.where(boundary_mask)[0]
    if len(boundary_idx) == 0:
        raise ValueError("未检测到边界点，请调小角度阈值或检查数据")

    first_boundary_point = points_3d[boundary_idx[0]]
    output = points_3d.clone()
    output[~boundary_mask] = first_boundary_point

    return output.unsqueeze(0)
