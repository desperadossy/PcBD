import numpy as np
import torch
import torch.nn as nn
from .Blocks import FeatureExtractor, BoundDetect
from .funcs import score_extract
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL1_PM
from extensions.emd.emd_module import emdModule

@MODELS.register_module()     
class PcBD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.FE = FeatureExtractor(norm_nsample = config.encoder_config.norm_nsample, knn = config.encoder_config.encoder_knn)
        self.BD = BoundDetect(embed_dim = config.decoder_config.embed_dim)
        self.build_loss_func()
    
    def build_loss_func(self):
        self.loss_func_CD = ChamferDistanceL1()
        self.loss_func_emd = emdModule()
        
    def get_loss(self, output, xyz, label, bound): 
        # data preparing 
        device = xyz.device
        nlabel = label[:,:,1].unsqueeze(-1)
        blabel = label[:,:,0].unsqueeze(-1)
        cxyz = score_extract(xyz.permute(0,2,1).contiguous(), nlabel.permute(0,2,1).contiguous(), reserve_high=False)
        bxyz = score_extract(cxyz, blabel.permute(0,2,1).contiguous(), reserve_high=True)
        noise_label, cleaned_xyz, bound_label, bound_xyz, moved_xyz = output
        # calculate loss
        criterion = nn.BCELoss().to(device)
        noise_label_loss = criterion(noise_label, nlabel)
        clean_point_loss = self.loss_func_CD(cleaned_xyz, cxyz.permute(0,2,1).contiguous())
        bound_label_loss = criterion(bound_label, blabel)
        bound_point_loss = self.loss_func_CD(bound_xyz, bxyz.permute(0,2,1).contiguous())
        move_loss = 1e4 * self.loss_func_CD(moved_xyz, bound)
        outlier_loss = (100 * noise_label_loss + 100 * clean_point_loss)
        bound_loss = (100 * bound_label_loss + 100 * bound_point_loss)
        return outlier_loss, bound_loss, move_loss
    
    def forward(self, xyz):
        """xyz: (B, N, 3)"""
        B, N, _ = xyz.size()
        output = []
        noise_label, cleaned_xyz, pw_feat = self.FE(xyz)
        bound_label, bound_xyz, moved_xyz = self.BD(cleaned_xyz, pw_feat)
        output.append(noise_label.permute(0,2,1).contiguous())
        output.append(cleaned_xyz.permute(0,2,1).contiguous())
        output.append(bound_label.permute(0,2,1).contiguous())
        output.append(bound_xyz.permute(0,2,1).contiguous())
        output.append(moved_xyz.permute(0,2,1).contiguous())
        return output

