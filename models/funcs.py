# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:32:09 2023

@author: 11517
"""
import math
import torch
import torch.nn as nn
from torch import einsum
from torch.autograd import Variable
from softmax_one.softmax_one import softmax_one
from .PAConv import knn_PAConv, knn_PAConv_with_norm
from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation


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
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)
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

    
def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz
    
    
def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd


class ProjectToXY(torch.autograd.Function):
    """Set the axisZ of a point cloud to zero"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clone()
        output[..., 2] = 0  # set axis-Z to zero
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[..., 2] = 0  # set the grads of Z to zero
        return grad_input
projectZ = ProjectToXY.apply


class ClearZ(torch.autograd.Function):
    """Clear the axisZ of the input point cloud"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input[:, :2, :].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_input[:, :2, :] = grad_output
        return grad_input
clearZ = ClearZ.apply


class GroupClearZ(torch.autograd.Function):
    """Clear the axisZ of the input kNN-Sampled point cloud"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input[:, :2, :, :].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_input[:, :2, :, :] = grad_output
        return grad_input
groupclearZ = GroupClearZ.apply


def knn_cal_axis(subset, use_XY=False):
    """Calculate point-wise 3-axis as SHOT"""
    B, N, k, _ = subset.size()
    subset_flipped = subset.permute(0,1,3,2).contiguous() #[B, N, 3, k]
    center = subset[:, :, :1, :]
    distance = torch.norm(subset, p=2, dim=-1) #[B, N, k]
    R = torch.max(distance, -1)[0].unsqueeze(-1) #[B, N, 1]
    local_weight = torch.exp(- distance / R).unsqueeze(-1) #[B, N, k, 1]
    cov_matrix = Variable(torch.matmul(subset_flipped, subset * local_weight), requires_grad=True)
    eigenvectors = torch.linalg.eigh(cov_matrix, UPLO='U')[1]
    Z_axis = eigenvectors[:, :, :, 2].unsqueeze(2) #calculate SVD [B, N, 1, 3]
    ambigue_judge_Z = torch.sum(torch.matmul(-Z_axis, subset_flipped).squeeze(2), dim=2)
    mask_Z = (ambigue_judge_Z < 0).float().unsqueeze(-1)
    Z_axis = Z_axis.squeeze(2) * (1 - 2*mask_Z)
    Z_axis = l2_norm(Z_axis, dim=-1)
    if use_XY:
        X_axis = eigenvectors[:, :, 0].unsqueeze(2) # [B, N, 1, 3]
        ambigue_judge_X = torch.sum(torch.matmul(-X_axis, subset_flipped).squeeze(2), dim=2)
        mask_X = (ambigue_judge_X < 0).float().unsqueeze(-1)
        X_axis = X_axis.squeeze(2) * (1 - 2*mask_X)
        X_axis = l2_norm(X_axis, dim=-1)
        Y_axis = torch.cross(Z_axis,X_axis, dim=-1)
        return Z_axis.permute(0,2,1).contiguous(), X_axis.permute(0,2,1).contiguous(), Y_axis.permute(0,2,1).contiguous()
    else:
        return Z_axis.permute(0,2,1).contiguous()
    

def knn_cal_axis_pca(subset, use_XY=False):
    """
    Estimate per-point normal (Z axis) using standard PCA on local neighborhood.
    Input:
        subset: [B, N, k, 3]  # local neighbor coordinates (centered or relative)
        use_XY: bool          # whether to compute full local frame
    Output:
        Z_axis: [B, 3, N]
        (if use_XY) also return X_axis and Y_axis, all in shape [B, 3, N]
    """
    B, N, k, _ = subset.size()
    # center the neighborhood
    mean = subset.mean(dim=2, keepdim=True)  # [B, N, 1, 3]
    centered = subset - mean  # [B, N, k, 3]

    # compute covariance matrix: [B, N, 3, 3]
    subset_flipped = centered.permute(0, 1, 3, 2).contiguous()  # [B, N, 3, k]
    cov_matrix = torch.matmul(subset_flipped, centered) / k  # [B, N, 3, 3]

    # eigen decomposition
    eigenvectors = torch.linalg.eigh(cov_matrix, UPLO='U')[1]  # [B, N, 3, 3]
    Z_axis = eigenvectors[:, :, :, 0]  # [B, N, 3] (smallest eigenvalue)
    Z_axis = l2_norm(Z_axis, dim=-1)

    # fix direction (optional - similar logic to your version)
    ambigue_judge_Z = torch.sum(Z_axis.unsqueeze(2) * subset, dim=-1).sum(-1)  # [B, N]
    mask_Z = (ambigue_judge_Z < 0).float().unsqueeze(-1)
    Z_axis = Z_axis * (1 - 2 * mask_Z)
    Z_axis = l2_norm(Z_axis, dim=-1)  # [B, N, 3]
    Z_axis = Z_axis.permute(0, 2, 1).contiguous()  # [B, 3, N]

    if use_XY:
        X_axis = eigenvectors[:, :, :, 2]  # [B, N, 3]
        ambigue_judge_X = torch.sum(X_axis.unsqueeze(2) * subset, dim=-1).sum(-1)
        mask_X = (ambigue_judge_X < 0).float().unsqueeze(-1)
        X_axis = X_axis * (1 - 2 * mask_X)
        X_axis = l2_norm(X_axis, dim=-1)
        Y_axis = torch.cross(Z_axis.permute(0, 2, 1), X_axis, dim=-1)
        X_axis = X_axis.permute(0, 2, 1).contiguous()
        Y_axis = Y_axis.permute(0, 2, 1).contiguous()
        return Z_axis, X_axis, Y_axis
    else:
        return Z_axis
        
def calculate_angles(grouped_xyz, grouped_norm, grouped_X, grouped_Y):
    B, _, N, k = grouped_xyz.shape
    
    # 计算与x轴方向的点乘
    dot_product_x = torch.sum(grouped_xyz * grouped_X, dim=1)  # B x N x k
    norm_neighbors = torch.norm(grouped_xyz, dim=1)  # B x N x k
    norm_x_vectors = torch.norm(grouped_X, dim=1)  # B x N x k
    cos_theta_x = dot_product_x / (norm_neighbors * norm_x_vectors + 1e-8)  # 避免除零
    angles_x = torch.acos(torch.clamp(cos_theta_x, -1.0, 1.0)) / math.pi  # B x N x k
    # 计算与y轴方向的点乘
    dot_product_y = torch.sum(grouped_xyz * grouped_Y, dim=1)  # B x N x k
    norm_y_vectors = torch.norm(grouped_Y, dim=1)  # B x N x k
    cos_theta_y = dot_product_y / (norm_neighbors * norm_y_vectors + 1e-8)
    angles_y = torch.acos(torch.clamp(cos_theta_y, -1.0, 1.0)) / math.pi  # B x N x k
    # 计算与z轴方向的点乘
    dot_product_z = torch.sum(grouped_xyz * grouped_norm, dim=1)  # B x N x k
    norm_z_vectors = torch.norm(grouped_norm, dim=1)  # B x N x k
    cos_theta_z = dot_product_z / (norm_neighbors * norm_z_vectors + 1e-8)
    angles_z = torch.acos(torch.clamp(cos_theta_z, -1.0, 1.0)) / math.pi  # B x N x k
    # 将三个轴的结果合并，形状为 B x N x 3 x k
    angles = torch.stack([angles_z, angles_x, angles_y], dim=1)  # B x 3 x N x k 
    return angles


class MLP(nn.Module):
    def __init__(self, in_channel, layer_dims, if_bn=None, activation_fn=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if if_bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(activation_fn)
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_2D(nn.Module):
    def __init__(self, in_channel, layer_dims, if_bn=None, activation_fn=nn.ReLU()):
        super(MLP_2D, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            if if_bn:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(activation_fn)
            last_channel = out_channel
        layers.append(nn.Conv2d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, if_bn=False, activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out

    
class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=False, activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn
        
    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out
    
    
class PointNet_SA_Module_KNN(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, if_bn=True, group_all=False, use_xyz=False, if_idx=False):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module_KNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3
        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(knn_PAConv_with_norm(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(knn_PAConv_with_norm(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, norm, X_axis, Y_axis, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            norm: Tensor, (B, 3, N)
            points: Tensor, (B, C, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_norm: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
            grouped_norm = norm.unsqueeze(2)
            new_norm = norm
        else:
            new_xyz, new_points, idx, grouped_xyz, grouped_dist = sample_and_group_knn(xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx, use_dist=True)
            grouped_norm = grouping_operation(norm, idx)
            grouped_X = grouping_operation(X_axis, idx)
            grouped_Y = grouping_operation(Y_axis, idx)
            grouped_angles = calculate_angles(grouped_xyz, grouped_norm, grouped_X, grouped_Y) # [B, 3, N, k]
            new_norm = grouped_norm[:,:,:,:1].squeeze(-1).contiguous()
            new_X = grouped_X[:,:,:,:1].squeeze(-1).contiguous()
            new_Y = grouped_Y[:,:,:,:1].squeeze(-1).contiguous()
        for conv in self.mlp_conv:
            new_points = conv(grouped_xyz, grouped_dist, grouped_norm, grouped_angles, new_points)
        new_points = torch.max(new_points, 3)[0]
        if self.if_idx:
            return new_xyz, new_norm, new_X, new_Y, new_points, idx
        else:
            return new_xyz, new_norm, new_X, new_Y, new_points


class PointNet_FP_Module(nn.Module):
    def __init__(self, in_channel, mlp, use_points1=True, in_channel_points1=None, if_bn=True):
        """
        Args:
            in_channel: int, input channel of points2
            mlp: list of int
            use_points1: boolean, if use points
            in_channel_points1: int, input channel of points1
        """
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1
        if use_points1:
            in_channel += in_channel_points1
        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz1, xyz2, norm1, norm2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, 3, N)
            xyz2: Tensor, (B, 3, M)
            norm1: Tensor, (B, 3, N)
            norm2: Tensor, (B, 3, M)
            points1: Tensor, (B, in_channel, N)
            points2: Tensor, (B, in_channel, M)
        Returns:
            new_points: Tensor, (B, mlp[-1], N)
        """
        B, _, N = xyz1.size()
        dist, idx = three_nn(xyz1.permute(0, 2, 1).contiguous(), xyz2.permute(0, 2, 1).contiguous()) # [B, N, 3]
        grouped_norm = grouping_operation(norm2, idx) # [B, 3, N, 3]
        norm_dist = torch.norm((norm1.unsqueeze(-1).repeat(1,1,1,3).permute(0,2,3,1).contiguous().reshape(B*N*3, -1) 
                   - grouped_norm.permute(0,2,3,1).contiguous().reshape(B*N*3, -1)), dim=-1).reshape(B, N, 3) # [B, N, 3]
        dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        recip_dist = 1.0/dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
        norm_dist = torch.clamp_min(norm_dist, 1e-10)  # (B, N, 3)
        recip_norm_dist = 1.0/norm_dist
        norm_norm = torch.sum(recip_norm_dist, 2, keepdim=True).repeat((1, 1, 3))
        norm_weight = recip_norm_dist / norm_norm
        weight = weight * norm_weight
        interpolated_points = three_interpolate(points2, idx, weight) #[B, E, N]
        if self.use_points1:
            new_points = torch.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points
        new_points = self.mlp_conv(new_points)
        return new_points
            

class Transformer_with_norm(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=8, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer_with_norm, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)
        self.pos_mlp = MLP_2D(3, [pos_hidden_dim, dim], if_bn=True)
        self.norm_mlp = MLP_2D(3, [pos_hidden_dim, dim], if_bn=True)
        self.xyz_mlp = nn.Conv2d(dim, dim, 1)
        self.attn_mlp = MLP_2D(dim, [dim * attn_hidden_multiplier, dim], if_bn=True)
        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos, norm):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)
            norm: Tensor of positions, (B, 3, n)
        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """
        identity = x
        x = self.linear_start(x)
        b, dim, n = x.shape
        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)
        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        norm_rel = norm.reshape((b, -1, n, 1)) - grouping_operation(norm, idx_knn)
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn
        norm_embedding = self.norm_mlp(norm_rel)
        xyz_embedding = self.xyz_mlp(pos_embedding + norm_embedding)
        attention = self.attn_mlp(qk_rel + xyz_embedding)
        attention = softmax_one(attention, -1)
        value = value.reshape((b, -1, n, 1)) + xyz_embedding
        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)
        return y+identity


class Transformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4, pc_dim=3):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)
        self.pos_mlp = MLP_2D(pc_dim, [pos_hidden_dim, dim], if_bn=True)
        self.attn_mlp = MLP_2D(dim, [dim * attn_hidden_multiplier, dim], if_bn=True)
        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """
        identity = x
        x = self.linear_start(x)
        b, dim, n = x.shape
        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)
        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn
        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = softmax_one(attention, -1)
        value = value.reshape((b, -1, n, 1)) + pos_embedding
        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)
        return y+identity
        


class Project_Transformer(nn.Module):
    def __init__(self, in_channel, dim=128, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Project_Transformer, self).__init__()
        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        self.pos_mlp = MLP_2D(2, [pos_hidden_dim, dim])
        self.xyz_mlp = MLP_2D(3, [pos_hidden_dim, dim])
        self.attn_mlp = MLP_2D(dim, [dim * attn_hidden_multiplier, dim])
        self.xyz_attn_mlp = MLP_2D(dim, [dim * attn_hidden_multiplier, dim])
        self.conv_kq = nn.Conv1d(dim, in_channel, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, xyz, pos, key, query, include_self=True):
        """
        Args:
            xyz: (B, 3, N)
            pos: (B, 2, N)
            key: (B, C, N)
            query: (B, C, N)
            include_self: boolean
        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        value = self.mlp_v(torch.cat([key, query], 1)) # [B, C, N]
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        value = self.conv_value(value)
        B, _, N = value.shape
        
        xyz_flipped = xyz.permute(0, 2, 1).contiguous() # [B, N, 3]
        xyz_idx_knn = query_knn(self.n_knn, xyz_flipped, xyz_flipped, include_self=include_self)
        grouped_query = grouping_operation(query, xyz_idx_knn)  # [B, D, N, k]
        kq_rel = key.reshape((B, -1, N, 1)) - grouped_query
        xyz_rel = xyz.reshape((B, -1, N, 1)) - grouping_operation(xyz, xyz_idx_knn)  # [B, 3, N, k]
        xyz_embedding = self.xyz_mlp(xyz_rel)
        xyz_attention = self.xyz_attn_mlp(kq_rel + xyz_embedding)  # [B, D, N, k]
        xyz_attention = softmax_one(xyz_attention, -1)
        value = value.unsqueeze(-1) + xyz_embedding 
        agg = einsum('b c i j, b c i j -> b c i', xyz_attention, value)  # [B, C, N]
        value = self.conv_kq(agg) + identity
        value = self.conv_value(value)
        
        pos_flipped = pos.permute(0, 2, 1).contiguous() # [B, N, 2]
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped, include_self=include_self)
        grouped_key = grouping_operation(key, idx_knn)  # [B, D, N, k]
        qk_rel = query.reshape((B, -1, N, 1)) - grouped_key
        pos_rel = pos.reshape((B, -1, N, 1)) - grouping_operation(pos, idx_knn)  # [B, 2, N, k]
        pos_embedding = self.pos_mlp(pos_rel)
        attention = self.attn_mlp(qk_rel + pos_embedding)  # [B, D, N, k]
        attention = softmax_one(attention, -1)
        value = value.reshape((B, -1, N, 1)) + pos_embedding  #
        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.conv_end(agg)

        return y + identity

"""
def score_extract(points, scores, threshold=0.1, reserve_high=False):
    B, _, N = points.size()
    updated_points = points.clone()
    high_score_mask = scores > threshold
    for b in range(B):
        high_score_points = points[b, :, high_score_mask[b, 0]].unsqueeze(0).permute(0, 2, 1).contiguous()
        low_score_points = points[b, :, ~high_score_mask[b, 0]].unsqueeze(0).permute(0, 2, 1).contiguous()
        if ((high_score_points.size(1)>2) & (low_score_points.size(1)>2)):
            if (~reserve_high):
                # 使用 query_knn 查找最近邻点
                knn_idx = query_knn(2, low_score_points, high_score_points, include_self=False).squeeze()
                # 由于 nsample=1，所以每个高评分点只有一个最近邻点
                for i, idx in enumerate(knn_idx[:, 1]):
                    closest_point = low_score_points[0, idx]
                    high_score_idx = torch.where(high_score_mask[b, 0])[0][i]
                    updated_points[b, :, high_score_idx] = closest_point
            else:
                knn_idx = query_knn(2, high_score_points, low_score_points, include_self=False).squeeze()
                for i, idx in enumerate(knn_idx[:, 1]):
                    closest_point = high_score_points[0, idx]
                    low_score_idx = torch.where(~high_score_mask[b, 0])[0][i]
                    updated_points[b, :, low_score_idx] = closest_point
    return updated_points"""

def score_extract(points, scores, threshold=0.1, reserve_high=False):
    B, _, N = points.size()
    if reserve_high:
        high_score_mask = scores > threshold
    else:
        high_score_mask = scores < threshold
    #print(torch.max(scores))
    points = points * high_score_mask

    nonzero_index = torch.max(high_score_mask, -1, keepdim=True)[1]
    replace_index = gather_operation(points, nonzero_index.int())  # Shape: (B, 1)
    points = torch.where(points == 0, replace_index, points)  # Broadcasting replace_index
    return points