import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numpy as np
from .PAConv import knn_PAConv, knn_PAConv_with_norm
from softmax_one.softmax_one import softmax_one
import math
from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
from .funcs import sample_and_group_knn, MLP, Transformer, Transformer_with_norm, Project_Transformer, knn_cal_axis, knn_cal_axis_pca, PointNet_SA_Module_KNN, PointNet_FP_Module, score_extract
from .funcs import projectZ, clearZ, groupclearZ, query_knn, calculate_angles

    
class FeatureExtractor(nn.Module):
    def __init__(self, norm_nsample=10,  knn=16):
        """Encoder that encodes information of partial point cloud"""
        super(FeatureExtractor, self).__init__()
        #params
        self.norm_nsample = norm_nsample
        self.knn = knn
        #layers
        self.sa1 = PointNet_SA_Module_KNN(1024, self.knn, 6, [64, 128], group_all=False, if_bn=False)
        self.transformer1 = Transformer_with_norm(128, dim=64)
        self.sa2 = PointNet_SA_Module_KNN(256, self.knn, 128, [128, 256], group_all=False, if_bn=False)
        self.transformer2 = Transformer_with_norm(256, dim=64)
        self.fp1 = PointNet_FP_Module(in_channel=256, mlp=[256, 256], in_channel_points1=128, if_bn=False)
        self.fp2 = PointNet_FP_Module(in_channel=256, mlp=[128, 128], in_channel_points1=6, if_bn=False)
        self.out_mlp = MLP(128, [64, 8, 1], if_bn=False)

        
    def forward(self, xyz):
        """
        Args:
             xyz: B, N, 3

        Returns:
            l3_points: (B, out_dim, 1)
        """
        #CalNorm
        B, N, _ = xyz.size()
        xyz_flipped = xyz.permute(0,2,1).contiguous() # [B, 3, N]
        _, _, idx, grouped_xyz = sample_and_group_knn(xyz_flipped.contiguous(), points=None, npoint=N, k=self.norm_nsample) # [B, 3, N, k], [B, N, k]
        norm, l0_X, l0_Y = knn_cal_axis(grouped_xyz.permute(0,2,3,1).contiguous(), use_XY=True) # [B, 3, N]
        pos = torch.cat((xyz_flipped, norm), dim=1) #[B, 6, N]
        #SA
        l0_points = pos.contiguous()
        l0_xyz = pos[:,:3,:].contiguous()
        l0_norm = pos[:,3:,:].contiguous()
        l0_X = l0_X.contiguous()
        l0_Y = l0_Y.contiguous()
        l1_xyz, l1_norm, l1_X, l1_Y, l1_points = self.sa1(l0_xyz, l0_norm, l0_X, l0_Y, l0_points) #[B, 3, N1], [B, 3, N1], [B, 128, N1]
        l1_points = self.transformer1(l1_points, l1_xyz, l1_norm) #[B, 128, N1]
        l2_xyz, l2_norm, l2_X, l2_Y, l2_points = self.sa2(l1_xyz, l1_norm, l1_X, l1_Y, l1_points) #[B, 3, N2], [B, 3, N2], [B, 256, N2]
        l2_points = self.transformer2(l2_points, l2_xyz, l2_norm) #[B, 256, N2]
        #FP
        l1_points = self.fp1(l1_xyz, l2_xyz, l1_norm, l2_norm, l1_points, l2_points) # [B, 256, N1]
        l0_points = self.fp2(l0_xyz, l1_xyz, l0_norm, l1_norm, l0_points, l1_points) # [B, 128, N]
        noise_label = torch.sigmoid(self.out_mlp(l0_points)) # [B, 1, N]
        #Clear outliers
        cleaned_xyz = score_extract(xyz_flipped, noise_label, threshold=0.7, reserve_high=False) # [B, 3, N]
        #cleaned_norm = score_extract(norm, noise_label, threshold=0.7, reserve_high=False) # [B, 3, N]
        pw_feat = score_extract(l0_points, noise_label, threshold=0.7, reserve_high=False) # [B, 128, N]
        return noise_label, cleaned_xyz, pw_feat


    
class BoundDetect(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed = nn.Conv2d(2, self.embed_dim, 1)
        self.project_embed = knn_PAConv(self.embed_dim, self.embed_dim, if_bn = False, dimension=2)
        self.merge_mlp = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=16)
        self.PT = Project_Transformer(self.embed_dim, dim=64)
        self.PA = Transformer(self.embed_dim, dim=64, pc_dim=2)
        self.label_pred =  MLP(self.embed_dim, [64, 8, 1], if_bn=False)
        self.bound_embed = knn_PAConv(2*self.embed_dim, self.embed_dim, if_bn = False, dimension=3)
        self.BoundPA = Transformer(8*self.embed_dim, dim=self.embed_dim, pc_dim=2)
        self.vec_mlp = MLP(8*self.embed_dim, [self.embed_dim, 8, 3], if_bn=False)
        
        
    def forward(self, xyz, pw_feat):
        """
        Args:
             xyz: B, 3, N
             pw_feat: B, C, N
        Returns:
            l3_points: (B, out_dim, 1)
        """
        B, _, N = xyz.size()
        #project
        projected_xyz = projectZ(xyz) # [B, 3, N], Z=0
        _, _, _, grouped_projected_xyz, grouped_dist = sample_and_group_knn(projected_xyz.contiguous(), points=None, npoint=N, k=16, use_dist=True) # [B, 3, N, k], [B, N, k]
        grouped_projected_xyz = groupclearZ(grouped_projected_xyz) # [B, 2, N, k]
        projected_xyz = clearZ(projected_xyz) # [B, 2, N]
        projected_feat = grouped_projected_xyz/ (1+grouped_dist.unsqueeze(1)) # [B, 2, N, k]
        projected_feat = self.project_embed(grouped_projected_xyz, self.pos_embed(projected_feat)).permute(0, 2, 1, 3).contiguous().view(-1, self.embed_dim, 16) # [B, E, N, k]
        projected_feat = self.merge_mlp(projected_feat).view(B, N, self.embed_dim).permute(0, 2, 1).contiguous() # [B, E, N]
        #bound
        projected_feat = self.PA(projected_feat, projected_xyz) # [B, E, N]
        projected_feat = self.PT(xyz, projected_xyz, pw_feat, projected_feat) # [B, E, N]
        bound_label = torch.sigmoid(self.label_pred(projected_feat)) # [B, 1, N]
        bound_xyz = score_extract(xyz, bound_label, threshold=0.5, reserve_high=True) # [B, 3, N]
        projected_bound = clearZ(bound_xyz) # [B, 2, N]
        # move
        knn_idx = query_knn(8, projected_xyz.permute(0,2,1).contiguous(), projected_bound.permute(0,2,1).contiguous())
        #grouped_bound = grouping_operation(projected_xyz, knn_idx) # [B, 2, N, k]
        grouped_xyz = grouping_operation(xyz, knn_idx) # [B, 3, N, k]
        bound_feat = grouping_operation(projected_feat, knn_idx) # [B, E, N, k]
        bound_xyz_feat = grouping_operation(pw_feat, knn_idx) # [B, E, N, k]
        bound_feat = torch.cat((bound_feat, bound_xyz_feat), dim=1) # [B, 2E, N, k]
        bound_feat = self.bound_embed(grouped_xyz, bound_feat).permute(0,3,1,2).contiguous().reshape(B, -1, N) # [B, kE, N]
        move_vec = self.BoundPA(bound_feat, projected_bound)
        move_vec = self.vec_mlp(bound_feat) # [B, 2, N]"""
        moved_xyz = bound_xyz + move_vec/10.0
        return bound_label, bound_xyz, moved_xyz 