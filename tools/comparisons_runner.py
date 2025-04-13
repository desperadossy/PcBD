import torch
import torch.nn as nn
import os
import sys
import json
import importlib
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

def comparisons_run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    if "scheduler" in config:
        scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)

    # trainval 
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['model_loss', 'BoundLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            points = data[0].cuda()
            labels = data[1].cuda()
            gt_bound = data[2].cuda()
            num_iter += 1
            if (config.model.NAME == "ScoreDeNoise"):
                model_loss, bound_loss = base_model.module.get_loss(points, labels, gt_bound)
            else:
                output = base_model(points)
                model_loss, bound_loss = base_model.module.get_loss(output, points, labels, gt_bound)
            _loss = model_loss + bound_loss
            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                model_loss = dist_utils.reduce_tensor(model_loss, args)
                bound_loss = dist_utils.reduce_tensor(bound_loss, args)
                losses.update([model_loss.item(), bound_loss.item()])
            else:
                losses.update([model_loss.item(), bound_loss.item()])

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/model', model_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Bound', bound_loss.item(), n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
            if "scheduler" in config:
                if config.scheduler.type == 'GradualWarmup':
                    if n_itr < config.scheduler.kwargs_2.total_epoch:
                        scheduler.step()
        if "scheduler" in config:
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step()
            else:
                scheduler.step()
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/model', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['Outlier_loss_l1', 'Outlier_loss_l2', 'Move_loss_l1', 'Move_loss_l2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    interval =  n_samples // 10

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            points = data[0].cuda()
            labels = data[1].cuda()
            gt_bound = data[2].cuda()
            nlabel = labels[:,:,1].unsqueeze(-1)
            blabel = labels[:,:,0].unsqueeze(-1)
            cxyz = score_extract(points.permute(0,2,1).contiguous(), nlabel.permute(0,2,1), reserve_high=False).permute(0,2,1)
            
            
            output = base_model(points)
            boundary = score_extract(output.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)

            Outlier_loss_l1 =  ChamferDisL1(output, cxyz)
            Outlier_loss_l2 =  ChamferDisL2(output, cxyz)
            Move_loss_l1 =  ChamferDisL1(boundary, gt_bound)
            Move_loss_l2 =  ChamferDisL2(boundary, gt_bound)
    
            if args.distributed:
                Outlier_loss_l1 = dist_utils.reduce_tensor(Outlier_loss_l1, args)
                Outlier_loss_l2 = dist_utils.reduce_tensor(Outlier_loss_l2, args)
                Move_loss_l1 = dist_utils.reduce_tensor(Move_loss_l1, args)
                Move_loss_l2 = dist_utils.reduce_tensor(Move_loss_l2, args)

            test_losses.update([Outlier_loss_l1.item() * 1000, Outlier_loss_l2.item() * 1000, Move_loss_l1.item() * 1000, Move_loss_l2.item() * 1000])

            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(boundary, gt_bound)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)


            # if val_writer is not None and idx % 200 == 0:
            #     input_pc = partial.squeeze().detach().cpu().numpy()
            #     input_pc = misc.get_ptcloud_img(input_pc)
            #     val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

            #     sparse = coarse_points.squeeze().cpu().numpy()
            #     sparse_img = misc.get_ptcloud_img(sparse)
            #     val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

            #     dense = dense_points.squeeze().cpu().numpy()
            #     dense_img = misc.get_ptcloud_img(dense)
            #     val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
                
            #     gt_ptcloud = gt.squeeze().cpu().numpy()
            #     gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
            #     val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')
        
            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def comparisons_test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    if config.model.NAME == "ScoreDeNoise" or config.model.NAME == "DMRDeNoise" :
        base_model = builder.model_builder(config.model)
        builder.load_model(base_model, args.ckpts, logger = logger)
        if args.use_gpu:
            base_model.to(args.local_rank)
    else:
        TEST_PATH = os.path.join(BASE_DIR, 'testmodels', config.model.NAME)
        sys.path.append(TEST_PATH)
        MODEL = importlib.import_module(config.model.NAME)
        Network = getattr(MODEL, config.model.NAME)
        base_model = Network
        

    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):
    if config.model.NAME == "ROR" or config.model.NAME == "SOR" or config.model.NAME == "DBSCAN":
        test_losses = AverageMeter(['Outlier_loss_l1', 'Outlier_loss_l2'])
        test_metrics = {
        "Outlier": AverageMeter(Metrics.names())
        }
    elif config.model.NAME == "AlphaShapes" or config.model.NAME == "AdaAlphaShapes" or config.model.NAME == "GridContour" or config.model.NAME == "NormalComparing":
        test_losses = AverageMeter(['Bound_loss_l1', 'Bound_loss_l2'])
        test_metrics = {
        "Boundary": AverageMeter(Metrics.names())
        }
    elif config.model.NAME == "BilateralFilter" or config.model.NAME == "IterNormFilter" or config.model.NAME == "W_MultiProj" or config.model.NAME == "MLS" or config.model.NAME == "AdaMLS" or config.model.NAME == "SparseReg":
        test_losses = AverageMeter(['Move_loss_l1', 'Move_loss_l2'])
        test_metrics = {
        "Smoothing": AverageMeter(Metrics.names())
        }
    else:
        base_model.eval()  # set model to eval mode
        test_losses = AverageMeter(['Outlier_loss_l1', 'Outlier_loss_l2', 'Move_loss_l1', 'Move_loss_l2'])
        test_metrics = {
        "Outlier": AverageMeter(Metrics.names()),
        "Smoothing": AverageMeter(Metrics.names())
        }
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
            if config.model.NAME == "AlphaShapes" or config.model.NAME == "AdaAlphaShapes" or config.model.NAME == "GridContour" or config.model.NAME == "NormalComparing":
                output = base_model(cxyz)
            elif config.model.NAME == "BilateralFilter" or config.model.NAME == "IterNormFilter" or config.model.NAME == "W_MultiProj" or config.model.NAME == "MLS" or config.model.NAME == "AdaMLS" or config.model.NAME == "SparseReg":
                output = base_model(cxyz)
                boundary = score_extract(output.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
            else:
                output = base_model(points)
            if config.model.NAME == "ROR" or config.model.NAME == "SOR" or config.model.NAME == "DBSCAN":
                Outlier_loss_l1 =  ChamferDisL1(output, cxyz)
                Outlier_loss_l2 =  ChamferDisL2(output, cxyz)
                test_losses.update([Outlier_loss_l1.item() * 1000, Outlier_loss_l2.item() * 1000])
                _metrics_Outlier = Metrics.get(output, cxyz)  
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = {
                        "Outlier": AverageMeter(Metrics.names())
                    }
                category_metrics[taxonomy_id]["Outlier"].update(_metrics_Outlier)
                if (idx+1) % 200 == 0:
                    print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics (Outlier) = %s' %
                              (idx + 1, n_samples, taxonomy_id, model_id,
                               ['%.4f' % l for l in test_losses.val()], 
                               ['%.4f' % m for m in _metrics_Outlier]), 
                              logger=logger)
            elif config.model.NAME == "AlphaShapes" or config.model.NAME == "AdaAlphaShapes" or config.model.NAME == "GridContour" or config.model.NAME == "NormalComparing":
                bxyz = score_extract(cxyz.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
                Bound_loss_l1 =  ChamferDisL1(output, bxyz)
                Bound_loss_l2 =  ChamferDisL2(output, bxyz)
                test_losses.update([Bound_loss_l1.item() * 1000, Bound_loss_l2.item() * 1000])
                _metrics_Boundary = Metrics.get(output, bxyz)  
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = {
                        "Boundary": AverageMeter(Metrics.names())
                    }
                category_metrics[taxonomy_id]["Boundary"].update(_metrics_Boundary)
                if (idx+1) % 200 == 0:
                    print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics (Boundary) = %s' %
                              (idx + 1, n_samples, taxonomy_id, model_id,
                               ['%.4f' % l for l in test_losses.val()], 
                               ['%.4f' % m for m in _metrics_Boundary]), 
                              logger=logger)
            elif config.model.NAME == "BilateralFilter" or config.model.NAME == "IterNormFilter" or config.model.NAME == "W_MultiProj" or config.model.NAME == "MLS" or config.model.NAME == "AdaMLS" or config.model.NAME == "SparseReg":
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
                    print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics (Smoothing) = %s' %
                              (idx + 1, n_samples, taxonomy_id, model_id,
                               ['%.4f' % l for l in test_losses.val()],  
                               ['%.4f' % m for m in _metrics_Smoothing]), 
                              logger=logger)
            else:
                boundary = score_extract(output.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
                Outlier_loss_l1 =  ChamferDisL1(output, cxyz)
                Outlier_loss_l2 =  ChamferDisL2(output, cxyz)
                Move_loss_l1 =  ChamferDisL1(boundary, gt_bound)
                Move_loss_l2 =  ChamferDisL2(boundary, gt_bound)
                test_losses.update([Outlier_loss_l1.item() * 1000, Outlier_loss_l2.item() * 1000, Move_loss_l1.item() * 1000, Move_loss_l2.item() * 1000])
                _metrics_Outlier = Metrics.get(output, cxyz)  
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
        if config.model.NAME == "ROR" or config.model.NAME == "SOR" or config.model.NAME == "DBSCAN":
            for _, v in category_metrics.items():
                test_metrics["Outlier"].update(v["Outlier"].avg())
        elif config.model.NAME == "AlphaShapes" or config.model.NAME == "AdaAlphaShapes" or config.model.NAME == "GridContour" or config.model.NAME == "NormalComparing":
            for _, v in category_metrics.items():
                test_metrics["Boundary"].update(v["Boundary"].avg())
        elif config.model.NAME == "BilateralFilter" or config.model.NAME == "IterNormFilter" or config.model.NAME == "W_MultiProj" or config.model.NAME == "MLS" or config.model.NAME == "AdaMLS" or config.model.NAME == "SparseReg":
            for _, v in category_metrics.items():
                test_metrics["Smoothing"].update(v["Smoothing"].avg())
        else:
            for _, v in category_metrics.items():
                test_metrics["Outlier"].update(v["Outlier"].avg())
                test_metrics["Smoothing"].update(v["Smoothing"].avg())
    if config.model.NAME == "ROR" or config.model.NAME == "SOR" or config.model.NAME == "DBSCAN":
        print_log('[TEST] Metrics (Outlier) = %s' % (['%.4f' % m for m in test_metrics["Outlier"].avg()]), logger=logger)
    elif config.model.NAME == "AlphaShapes" or config.model.NAME == "AdaAlphaShapes" or config.model.NAME == "GridContour" or config.model.NAME == "NormalComparing":
        print_log('[TEST] Metrics (Boundary) = %s' % (['%.4f' % m for m in test_metrics["Boundary"].avg()]), logger=logger)
    elif config.model.NAME == "BilateralFilter" or config.model.NAME == "IterNormFilter" or config.model.NAME == "W_MultiProj" or config.model.NAME == "MLS" or config.model.NAME == "AdaMLS" or config.model.NAME == "SparseReg":
        print_log('[TEST] Metrics (Smoothing) = %s' % (['%.4f' % m for m in test_metrics["Smoothing"].avg()]), logger=logger)
    else:
        print_log('[TEST] Metrics (Outlier) = %s' % (['%.4f' % m for m in test_metrics["Outlier"].avg()]), logger=logger)
        print_log('[TEST] Metrics (Smoothing) = %s' % (['%.4f' % m for m in test_metrics["Smoothing"].avg()]), logger=logger)

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    if config.model.NAME == "ROR" or config.model.NAME == "SOR" or config.model.NAME == "DBSCAN":
        print_log('====================== TEST RESULTS ======================', logger=logger)
        print_log('===============       Outlier_Removal       ==============', logger=logger)
    elif config.model.NAME == "AlphaShapes" or config.model.NAME == "AdaAlphaShapes" or config.model.NAME == "GridContour" or config.model.NAME == "NormalComparing":
        print_log('====================== TEST RESULTS =======================', logger=logger)
        print_log('==============       Boundary_Extract       ===============', logger=logger)
    elif config.model.NAME == "BilateralFilter" or config.model.NAME == "IterNormFilter" or config.model.NAME == "W_MultiProj" or config.model.NAME == "MLS" or config.model.NAME == "AdaMLS" or config.model.NAME == "SparseReg":
        print_log('====================== TEST RESULTS ======================', logger=logger)
        print_log('===============          Smoothing          ==============', logger=logger)
    else:
        print_log('====================================== TEST RESULTS ======================================', logger=logger)
        print_log('===============       Outlier_Removal       ===          Smoothing          ==============', logger=logger)
    msg = 'Taxonomy\t#Num '
    if config.model.NAME == "ROR" or config.model.NAME == "SOR" or config.model.NAME == "DBSCAN":
        for metric in test_metrics["Outlier"].items:
            msg += metric + '\t'
            if metric =='FS':
                msg += '\t'
    elif config.model.NAME == "AlphaShapes" or config.model.NAME == "AdaAlphaShapes" or config.model.NAME == "GridContour" or config.model.NAME == "NormalComparing":
        for metric in test_metrics["Boundary"].items:
            msg += metric + '\t'
            if metric =='FS':
                msg += '\t'
    elif config.model.NAME == "BilateralFilter" or config.model.NAME == "IterNormFilter" or config.model.NAME == "W_MultiProj" or config.model.NAME == "MLS" or config.model.NAME == "AdaMLS" or config.model.NAME == "SparseReg":
        for metric in test_metrics["Smoothing"].items:
            msg += metric + '\t'
            if metric =='FS':
                msg += '\t'
    else:
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
        if config.model.NAME == "ROR" or config.model.NAME == "SOR" or config.model.NAME == "DBSCAN":
            total_count += category_metrics[taxonomy_id]["Outlier"].count(0)
            msg += str(category_metrics[taxonomy_id]["Outlier"].count(0)) + '\t'
        elif config.model.NAME == "AlphaShapes" or config.model.NAME == "AdaAlphaShapes" or config.model.NAME == "GridContour" or config.model.NAME == "NormalComparing":
            total_count += category_metrics[taxonomy_id]["Boundary"].count(0)
            msg += str(category_metrics[taxonomy_id]["Boundary"].count(0)) + '\t'
        else:
            total_count += category_metrics[taxonomy_id]["Smoothing"].count(0)
            msg += str(category_metrics[taxonomy_id]["Smoothing"].count(0)) + '\t'
        if config.model.NAME == "ROR" or config.model.NAME == "SOR" or config.model.NAME == "DBSCAN":
            for value in category_metrics[taxonomy_id]["Outlier"].avg():
                msg += '%.3f \t' % value
        elif config.model.NAME == "AlphaShapes" or config.model.NAME == "AdaAlphaShapes" or config.model.NAME == "GridContour" or config.model.NAME == "NormalComparing":
            for value in category_metrics[taxonomy_id]["Boundary"].avg():
                msg += '%.3f \t' % value
        elif config.model.NAME == "BilateralFilter" or config.model.NAME == "IterNormFilter" or config.model.NAME == "W_MultiProj" or config.model.NAME == "MLS" or config.model.NAME == "AdaMLS" or config.model.NAME == "SparseReg":
            for value in category_metrics[taxonomy_id]["Smoothing"].avg():
                msg += '%.3f \t' % value
        else:
            for value in category_metrics[taxonomy_id]["Outlier"].avg():
                msg += '%.3f \t' % value
            for value in category_metrics[taxonomy_id]["Smoothing"].avg():
                msg += '%.3f \t' % value

        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = 'Overall'
    msg += '  '
    msg += str(total_count) + '\t'
    if config.model.NAME == "ROR" or config.model.NAME == "SOR" or config.model.NAME == "DBSCAN":
        for value in test_metrics["Outlier"].avg():
            msg += '%.3f \t' % value
    elif config.model.NAME == "AlphaShapes" or config.model.NAME == "AdaAlphaShapes" or config.model.NAME == "GridContour" or config.model.NAME == "NormalComparing":
        for value in test_metrics["Boundary"].avg():
            msg += '%.3f \t' % value
    elif config.model.NAME == "BilateralFilter" or config.model.NAME == "IterNormFilter" or config.model.NAME == "W_MultiProj" or config.model.NAME == "MLS" or config.model.NAME == "AdaMLS" or config.model.NAME == "SparseReg":
        for value in test_metrics["Smoothing"].avg():
            msg += '%.3f \t' % value
    else:
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