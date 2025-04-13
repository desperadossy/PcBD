import torch
import faulthandler
faulthandler.enable()

def GridContour(points: torch.Tensor, gsp: float = 0.015):
    """
    Grid-based Alpha Shape contour detection based on laser viewing direction.

    Args:
        points (torch.Tensor): (1, N, 3) input point cloud
        theta (torch.Tensor): (3,) viewing direction vector
        gsp (float): grid spacing

    Returns:
        torch.Tensor: (1, N, 3) output with boundary points kept and non-boundary points replaced
    """
    theta=torch.tensor([0, 0, 1.0], device=points.device)
    assert points.ndim == 3 and points.shape[0] == 1
    device = points.device
    N = points.shape[1]
    points_3d = points[0]  # (N, 3)

    # Step 1: 投影到垂直 theta 的平面
    theta = theta / torch.norm(theta)
    z_axis = theta
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
    if torch.allclose(x_axis, z_axis):
        x_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
    x_axis = x_axis - torch.dot(x_axis, z_axis) * z_axis
    x_axis = x_axis / torch.norm(x_axis)
    y_axis = torch.cross(z_axis, x_axis, dim=0)  # ✅ 指定 dim=0

    R = torch.stack([x_axis, y_axis, z_axis], dim=1)  # (3, 3)
    points_proj = (points_3d @ R)[:, :2]  # (N, 2)

    # Step 2: 网格构建
    xmin, ymin = torch.min(points_proj, dim=0).values
    xmax, ymax = torch.max(points_proj, dim=0).values
    Nx = int(torch.ceil((xmax - xmin) / gsp).item())
    Ny = int(torch.ceil((ymax - ymin) / gsp).item())

    xi = (xmin - gsp) + torch.arange(Nx + 3, device=device) * gsp
    yj = (ymin - gsp) + torch.arange(Ny + 3, device=device) * gsp

    # ✅ 确保输入 bucketize 的是连续 Tensor
    xi_idx = torch.bucketize(points_proj[:, 0].contiguous(), xi)
    yj_idx = torch.bucketize(points_proj[:, 1].contiguous(), yj)

    # Step 3: 构建网格标记矩阵
    grid_flags = torch.full((Ny + 3, Nx + 3), -1, dtype=torch.int8, device=device)
    for k in range(N):
        i = xi_idx[k].item()
        j = yj_idx[k].item()
        if 0 <= i < Nx + 3 and 0 <= j < Ny + 3:
            grid_flags[j, i] = 1  # (row, col)

    # Step 4: 网格轮廓边界检测（找出边界格子）
    boundary_mask = torch.zeros(N, dtype=torch.bool, device=device)

    for j in range(1, Ny + 2):
        for i in range(Nx + 2):
            m1 = grid_flags[j - 1, i]
            m2 = grid_flags[j, i]
            if m1 * m2 == -1:
                cond = ((yj_idx == j) | (yj_idx == j - 1)) & (xi_idx == i)
                boundary_mask |= cond

    for i in range(1, Nx + 2):
        for j in range(Ny + 2):
            m1 = grid_flags[j, i - 1]
            m2 = grid_flags[j, i]
            if m1 * m2 == -1:
                cond = ((xi_idx == i) | (xi_idx == i - 1)) & (yj_idx == j)
                boundary_mask |= cond

    # Step 5: 替换非边界点为首个边界点
    boundary_idx = torch.where(boundary_mask)[0]
    if len(boundary_idx) == 0:
        raise ValueError("No boundary points found. Adjust gsp or check input.")

    first_boundary_point = points_3d[boundary_idx[0]]
    output = points_3d.clone()
    output[~boundary_mask] = first_boundary_point

    return output.unsqueeze(0)  # (1, N, 3)
