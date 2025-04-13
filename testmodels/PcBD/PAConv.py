# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 19:54:23 2023

@author: 11517
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from softmax_one.softmax_one import softmax_one
_init = nn.init.kaiming_normal_


def get_ed(x, y):
    ed = torch.norm(x - y, dim=-1).reshape(x.shape[0], 1)
    return ed

    
class knn_ScoreNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], last_bn=False):
        super(knn_ScoreNet, self).__init__()
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()
        hidden_unit = list() if hidden_unit is None else copy.deepcopy(hidden_unit)
        hidden_unit.append(out_channel)
        hidden_unit.insert(0, in_channel)
        for i in range(1, len(hidden_unit)):  # from 1st hidden to next hidden to last hidden
            self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1,
                                                   bias=False if i < len(hidden_unit) - 1 else not last_bn))
            self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[i]))

    def forward(self, xyz):
        # xyz : B*3*N*K
        B, _, N, K = xyz.size()
        scores = xyz
        for i, conv in enumerate(self.mlp_convs_hidden):
            if i < len(self.mlp_convs_hidden) - 1:
                scores = F.relu(self.mlp_bns_hidden[i](conv(scores)))
            else:  # if the output layer, no ReLU
                scores = conv(scores)
                if self.last_bn:
                    scores = self.mlp_bns_hidden[i](scores)
        scores = softmax_one(scores, dim=1)  # + 0.5  # B*m*N*K
        scores = scores.permute(0, 2, 3, 1)  # B*N*K*m
        return scores

    
class knn_PAConv(nn.Module):
    def __init__(self, input_channel, output_channel, M=8, if_bn=True, activation_fn=torch.relu, dimension=3):
        super(knn_PAConv, self).__init__()
        self.m = M
        """"ScoreNet"""
        self.scorenet = knn_ScoreNet(dimension+1, self.m, hidden_unit=[16], last_bn=False)
        """"WeightBank"""
        tensor1 = _init(torch.empty(self.m, input_channel, output_channel)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(input_channel, M * output_channel)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)
        self.if_bn = if_bn
        if self.if_bn:
            self.bn = nn.BatchNorm2d(output_channel)
        self.activation_fn = activation_fn

    def forward(self, grouped_xyz, feature):
        B, _, N, k = grouped_xyz.size()
        xyz_ed = torch.norm(grouped_xyz.permute(0,2,3,1).contiguous().reshape(B*N*k, -1), dim=-1).reshape(B, 1, N, k)
        xyz = torch.cat((grouped_xyz, xyz_ed), dim=1) #[B,4,N,k]
        Scores = self.scorenet(xyz)  # [B, N, k, M]
        out_feat = torch.matmul(feature.permute(0,2,3,1).contiguous(), self.weightbank).view(B, N, k, self.m, -1)  #[B, N, k, M, C]
        out_feat = torch.matmul(Scores.unsqueeze(-2), out_feat).view(B, N, k, -1).permute(0,3,1,2).contiguous() #[B, C, N, k]
        if self.if_bn:
            out_feat = self.bn(out_feat)
        if self.activation_fn is not None:
            out_feat = self.activation_fn(out_feat)
        return out_feat


class knn_PAConv_with_norm(nn.Module):
    def __init__(self, input_channel, output_channel, M=8, if_bn=True, activation_fn=torch.relu):
        super(knn_PAConv_with_norm, self).__init__()
        self.m = M
        """"ScoreNet"""
        self.scorenet = knn_ScoreNet(14, self.m, hidden_unit=[16], last_bn=False)
        """"WeightBank"""
        tensor1 = _init(torch.empty(self.m, input_channel, output_channel)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(input_channel, M * output_channel)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)
        self.if_bn = if_bn
        if self.if_bn:
            self.bn = nn.BatchNorm2d(output_channel)
        self.activation_fn = activation_fn

    def forward(self, grouped_xyz, grouped_dist, grouped_norm, grouped_angle, feature):
        B, _, N, k = grouped_xyz.size()
        center_norm = grouped_norm[..., :1].repeat(1, 1, 1, k)
        grouped_norm_diff = grouped_norm - center_norm  #[B, 3, N, k]
        norm_ed = get_ed(center_norm.permute(0,2,3,1).contiguous().reshape(B*N*k, -1),
                    grouped_norm.permute(0,2,3,1).contiguous().reshape(B*N*k, -1)).reshape(B, 1, N, k) #[B,1,N,k]
        xyz_ed = torch.norm(grouped_xyz.permute(0,2,3,1).contiguous().reshape(B*N*k, -1), dim=-1).reshape(B, 1, N, k)
        xyz = torch.cat((grouped_xyz, xyz_ed, grouped_norm, grouped_norm_diff, norm_ed, grouped_angle), dim=1) / (1+grouped_dist.unsqueeze(1)) #[B,11,N,k]
        Scores = self.scorenet(xyz)  # [B, N, k, M]
        out_feat = torch.matmul(feature.permute(0,2,3,1).contiguous(), self.weightbank).view(B, N, k, self.m, -1)  #[B, N, k, M, C]
        out_feat = torch.matmul(Scores.unsqueeze(-2), out_feat).view(B, N, k, -1).permute(0,3,1,2).contiguous() #[B, C, N, k]
        if self.if_bn:
            out_feat = self.bn(out_feat)
        if self.activation_fn is not None:
            out_feat = self.activation_fn(out_feat)
        return out_feat
