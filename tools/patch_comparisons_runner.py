import torch
import torch.nn as nn
import os
import json
import numpy as np
import sys
import importlib
import argparse
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from provider import *

def patch_comparisons_test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    TEST_PATH = os.path.join(BASE_DIR, 'testmodels', config.model.NAME)
    sys.path.append(TEST_PATH)
    if config.model.NAME == "PointCleanNet":
        MODEL = importlib.import_module(config.model.NAME)
        Network = getattr(MODEL, config.model.NAME)
        base_model_outlier = Network(output_dim=1)
        base_model_denoise = Network(output_dim=3)
        outlier_checkpoint = torch.load(os.path.join(BASE_DIR, "checkpoints", f"{config.model.NAME}_Outliers.pth"), weights_only=True)
        denoise_checkpoint = torch.load(os.path.join(BASE_DIR, "checkpoints", f"{config.model.NAME}_DeNoise.pth"), weights_only=True)
        base_model_outlier.load_state_dict(outlier_checkpoint)
        base_model_denoise.load_state_dict(denoise_checkpoint)
        if args.use_gpu:
            base_model_outlier.to(args.local_rank)
            base_model_denoise.to(args.local_rank)
    elif config.model.NAME == "PD_LTS_Heavy" or config.model.NAME == "PD_LTS_Light" :
        MODEL = importlib.import_module(config.model.NAME)
        Network = getattr(MODEL, config.model.NAME)
        if config.model.NAME == "PD_LTS_Heavy":
            base_model = nn.ModuleList()
            for i in range(3):
                base_model.append(Network().cuda())
        else:
            base_model = Network().cuda()
        checkpoint = torch.load(args.ckpts, weights_only=True)
        base_model.load_state_dict(checkpoint)
        if args.use_gpu:
            base_model.to(args.local_rank)
    else:
        MODEL = importlib.import_module(config.model.NAME)
        Network = getattr(MODEL, config.model.NAME)
        base_model = Network()
        checkpoint = torch.load(args.ckpts, weights_only=True)
        base_ckpt = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
        base_model.load_state_dict(base_ckpt)
        if args.use_gpu:
            base_model.to(args.local_rank)
    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    if config.model.NAME == "PointCleanNet":
        test(base_model_outlier, base_model_denoise, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)
    else:
        test(base_model, None, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, second_base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):
    if config.model.NAME == "PD_LTS_Heavy":
        for i in range(3):
            base_model[i].init_as_trained_state()
            base_model[i].eval()
    else:   
        base_model.eval()  # set model to eval mode
    if second_base_model is not None:
        second_base_model.eval()
        test_losses = AverageMeter(['Outlier_loss_l1', 'Outlier_loss_l2', 'Move_loss_l1', 'Move_loss_l2'])
        test_metrics = {"Outlier": AverageMeter(Metrics.names()), "Smoothing": AverageMeter(Metrics.names())}
    elif config.model.NAME == "PD_LTS_Light" or config.model.NAME == "PD_LTS_Heavy":
        test_losses = AverageMeter(['Outlier_loss_l1', 'Outlier_loss_l2', 'Move_loss_l1', 'Move_loss_l2'])
        test_metrics = {"Outlier": AverageMeter(Metrics.names()), "Smoothing": AverageMeter(Metrics.names())}
    else:
        test_losses = AverageMeter(['Move_loss_l1', 'Move_loss_l2'])
        test_metrics = {"Smoothing": AverageMeter(Metrics.names())}

    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]
            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME

            points = data[0].cuda()
            labels = data[1].cuda()
            gt_bound = data[2].cuda()
            nlabel = labels[:,:,1].unsqueeze(-1)
            blabel = labels[:,:,0].unsqueeze(-1)
            cxyz = score_extract(points.permute(0,2,1).contiguous(), nlabel.permute(0,2,1), reserve_high=False).permute(0,2,1)
            bxyz = score_extract(cxyz.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
            if second_base_model is not None:
                _, _, _, xyz_patches = sample_and_group_knn(points.permute(0,2,1).contiguous(), points=None, npoint=npoints, k=500) 
                xyz_patches = xyz_patches.permute(0, 3, 1, 2) #[B, k, 3, N]
                outlier_labels = torch.zeros((1, npoints, 1), device=points.device) 
                for i in range(npoints):
                    input_patch = xyz_patches[:, :, :, i].repeat(2,1,1)
                    with torch.no_grad():
                        output = base_model(input_patch)[0].reshape(1,1)
                        outlier_labels[:,i] = output
                cpoints = score_extract(points.permute(0,2,1).contiguous(), outlier_labels.permute(0,2,1), threshold=0.5, reserve_high=False).permute(0,2,1)
                _, _, _, xyz_patches = sample_and_group_knn(points.permute(0,2,1).contiguous(), points=None, npoint=npoints, k=500) 
                xyz_patches = xyz_patches.permute(0, 3, 1, 2) #[B, k, 3, N]
                cleaned_xyz = torch.zeros((1, npoints, 3), device=cpoints.device)
                for i in range(npoints):
                    input_patch = xyz_patches[:, :, :, i].repeat(2,1,1)
                    with torch.no_grad():
                        move_vec = second_base_model(input_patch)[0].reshape(1,3)
                    new_point = input_patch[0, 0, :].detach().clone().reshape(1,3)
                    new_point += move_vec.detach() * 0.005
                    cleaned_xyz[:, i, :] = new_point 
                boundary = score_extract(cleaned_xyz.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
                Outlier_loss_l1 =  ChamferDisL1(cpoints, cxyz)
                Outlier_loss_l2 =  ChamferDisL2(cpoints, cxyz)
                Move_loss_l1 =  ChamferDisL1(boundary, gt_bound)
                Move_loss_l2 =  ChamferDisL2(boundary, gt_bound)
                test_losses.update([Outlier_loss_l1.item() * 1000, Outlier_loss_l2.item() * 1000, Move_loss_l1.item() * 1000, Move_loss_l2.item() * 1000])
                _metrics_Outlier = Metrics.get(cpoints, cxyz)  
                _metrics_Smoothing = Metrics.get(boundary, gt_bound) 
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = {
                    "Outlier": AverageMeter(Metrics.names()),
                    "Smoothing": AverageMeter(Metrics.names())
                    }
                category_metrics[taxonomy_id]["Outlier"].update(_metrics_Outlier)
                category_metrics[taxonomy_id]["Smoothing"].update(_metrics_Smoothing)
                if (idx+1) % 200 == 0:
                    print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics (Outlier) = %s  Metrics (Smoothing) = %s' %
                              (idx + 1, n_samples, taxonomy_id, model_id,
                               ['%.4f' % l for l in test_losses.val()], 
                               ['%.4f' % m for m in _metrics_Outlier], 
                               ['%.4f' % m for m in _metrics_Smoothing]), 
                              logger=logger)
            elif config.model.NAME == "PD_LTS_Light" or config.model.NAME == "PD_LTS_Heavy":
                points = points.squeeze()
                if config.model.NAME == "PD_LTS_Heavy":
                    cpoints = patch_denoise(base_model, points, 1000, 3).unsqueeze(0)
                else:
                    cpoints = light_patch_denoise(base_model, points, 1000, 3).unsqueeze(0)
                boundary = score_extract(cpoints.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
                Outlier_loss_l1 =  ChamferDisL1(cpoints, cxyz)
                Outlier_loss_l2 =  ChamferDisL2(cpoints, cxyz)
                Move_loss_l1 =  ChamferDisL1(boundary, gt_bound)
                Move_loss_l2 =  ChamferDisL2(boundary, gt_bound)
                test_losses.update([Outlier_loss_l1.item() * 1000, Outlier_loss_l2.item() * 1000, Move_loss_l1.item() * 1000, Move_loss_l2.item() * 1000])
                _metrics_Outlier = Metrics.get(cpoints, cxyz)  
                _metrics_Smoothing = Metrics.get(boundary, gt_bound) 
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = {
                    "Outlier": AverageMeter(Metrics.names()),
                    "Smoothing": AverageMeter(Metrics.names())
                    }
                category_metrics[taxonomy_id]["Outlier"].update(_metrics_Outlier)
                category_metrics[taxonomy_id]["Smoothing"].update(_metrics_Smoothing)
                if (idx+1) % 200 == 0:
                    print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics (Outlier) = %s  Metrics (Smoothing) = %s' %
                              (idx + 1, n_samples, taxonomy_id, model_id,
                               ['%.4f' % l for l in test_losses.val()], 
                               ['%.4f' % m for m in _metrics_Outlier], 
                               ['%.4f' % m for m in _metrics_Smoothing]), 
                              logger=logger)
            else:
                _, _, _, xyz_patches = sample_and_group_knn(bxyz.permute(0,2,1).contiguous(), points=None, npoint=npoints, k=500) 
                xyz_patches = xyz_patches.permute(0, 3, 1, 2) #[B, k, 3, N]
                cleaned_xyz = torch.zeros((1, npoints, 3), device=bxyz.device) 
                for i in range(npoints):
                    input_patch = xyz_patches[:, :, :, i]
                    with torch.no_grad():
                        move_vec = base_model(input_patch)
                        _, noise_inv = pca_alignment(input_patch.squeeze().detach().cpu())
                        noise_inv = torch.from_numpy(noise_inv).float().cuda()
                        move_vec = torch.bmm(noise_inv.unsqueeze(0), move_vec.reshape(1,3,1)).squeeze(-1)
                    new_point = input_patch[:, 0, :].detach().clone()  
                    new_point += move_vec.detach() * 0.005
                    cleaned_xyz[:, i, :] = new_point 
                #boundary = score_extract(cleaned_xyz.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
                boundary = cleaned_xyz
                Move_loss_l1 =  ChamferDisL1(boundary, gt_bound)
                Move_loss_l2 =  ChamferDisL2(boundary, gt_bound)
                test_losses.update([Move_loss_l1.item() * 1000, Move_loss_l2.item() * 1000])
                _metrics_Smoothing = Metrics.get(boundary, gt_bound) 
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = {
                        "Smoothing": AverageMeter(Metrics.names())
                    }
                category_metrics[taxonomy_id]["Smoothing"].update(_metrics_Smoothing)
                if (idx+1) % 200 == 0:
                    print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s  Metrics (Smoothing) = %s' %
                              (idx + 1, n_samples, taxonomy_id, model_id,
                               ['%.4f' % l for l in test_losses.val()], 
                               ['%.4f' % m for m in _metrics_Smoothing]), 
                              logger=logger)
        if second_base_model is not None or config.model.NAME == "PD_LTS_Light" or config.model.NAME == "PD_LTS_Heavy":
            for _, v in category_metrics.items():
                test_metrics["Outlier"].update(v["Outlier"].avg())
        for _, v in category_metrics.items():
            test_metrics["Smoothing"].update(v["Smoothing"].avg())
    if second_base_model is not None or config.model.NAME == "PD_LTS_Light" or config.model.NAME == "PD_LTS_Heavy":
        print_log('[TEST] Metrics (Outlier) = %s' % (['%.4f' % m for m in test_metrics["Outlier"].avg()]), logger=logger)
    print_log('[TEST] Metrics (Smoothing) = %s' % (['%.4f' % m for m in test_metrics["Smoothing"].avg()]), logger=logger)

     

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    if second_base_model is not None or config.model.NAME == "PD_LTS_Light" or config.model.NAME == "PD_LTS_Heavy":
        print_log('====================================== TEST RESULTS ======================================', logger=logger)
        print_log('===============       Outlier_Removal       ===          Smoothing          ==============', logger=logger)
    else:
        print_log('====================== TEST RESULTS ======================', logger=logger)
        print_log('===============          Smoothing          ==============', logger=logger)
    msg = 'Taxonomy\t#Num '
    if second_base_model is not None or config.model.NAME == "PD_LTS_Light" or config.model.NAME == "PD_LTS_Heavy":
        for metric in test_metrics["Outlier"].items:
            msg += metric + '\t'
            if metric =='FS':
                msg += '\t'
        msg += '\t'
    for metric in test_metrics["Smoothing"].items:
        msg += metric + '\t'
        if metric =='FS':
            msg += '\t'
    msg += '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)
    total_count = 0
    for taxonomy_id in category_metrics:
        msg = taxonomy_id + '\t'
        total_count += category_metrics[taxonomy_id]["Smoothing"].count(0)
        msg += str(category_metrics[taxonomy_id]["Smoothing"].count(0)) + '\t'
        if second_base_model is not None or config.model.NAME == "PD_LTS_Light" or config.model.NAME == "PD_LTS_Heavy":
            for value in category_metrics[taxonomy_id]["Outlier"].avg():
                msg += '%.3f \t' % value
        for value in category_metrics[taxonomy_id]["Smoothing"].avg():
            msg += '%.3f \t' % value

        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = 'Overall'
    msg += '  '
    msg += str(total_count) + '\t'
    if second_base_model is not None or config.model.NAME == "PD_LTS_Light" or config.model.NAME == "PD_LTS_Heavy":
        for value in test_metrics["Outlier"].avg():
            msg += '%.3f \t' % value
    for value in test_metrics["Smoothing"].avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return



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