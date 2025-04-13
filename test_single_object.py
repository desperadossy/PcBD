from datasets.io import IO
import numpy as np
import os
from tqdm import tqdm
import inspect
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from provider import *
import time


def main():
    '''PREPARING'''
    parser = argparse.ArgumentParser(description="test parser args")
    parser.add_argument("--method", default="PcBD", help="test network/other approaches")
    parser.add_argument("--num_points", default=4096, help="number of input points")
    parser.add_argument("--patch_size", default=500, help="number of points in patch-based networks")
    parser.add_argument("--object", default="/root/PcBD/data/test/02691156/1a04e3eab45ca15dd86060f189eb133", help="object to test")
    parser.add_argument("--Boundary3D", default=False, help="Detect 3D Boundary")
    parser.add_argument("--memory", default=False, help="Output Memory Usage, only use with networks!")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    TEST_PATH = os.path.join(BASE_DIR, 'testmodels', args.method)
    sys.path.append(TEST_PATH)
    '''DATA'''
    if not args.object.endswith(".pcd"):
        xyz = IO.get(os.path.join(args.object, "00", "input.pcd")).astype(np.float32)  
        labels = IO.get(os.path.join(args.object, "00", "label.npy")).astype(np.float32)  
        xyz, labels = group_random_sample(xyz, labels, args.num_points)
        np.savetxt(os.path.join(TEST_PATH, "noisy.txt"), xyz)
        xyz = torch.from_numpy(xyz).float().unsqueeze(0).cuda()
    else:
        xyz = IO.get(args.object).astype(np.float32)  
        xyz = random_sample(pc_normalize(xyz), args.num_points)
        np.savetxt(os.path.join(TEST_PATH, "noisy.txt"), xyz)
        xyz = torch.from_numpy(xyz).float().unsqueeze(0).cuda()
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.method)
    Network = getattr(MODEL, args.method)
    if args.method == "PointCleanNet":
        classifier_outlier = Network(output_dim=1).cuda().eval()
        classifier_denoise = Network(output_dim=3).cuda().eval()
    elif args.method == "PD_LTS_Heavy":
        classifier = nn.ModuleList()
        for i in range(3):
            classifier.append(Network().cuda())
    else:
        if inspect.isclass(Network):
            classifier = Network().cuda().eval()
        else:
            classifier = Network
    #for key, value in checkpoint.items():
        #print(f"{key}: {type(value)}")
    if args.method == "PcBD"  or args.method == "ScoreDeNoise" or args.method == "DMRDeNoise":
        checkpoint = torch.load(os.path.join(BASE_DIR, "checkpoints", f"{args.method}.pth"), weights_only=True)
        base_ckpt = {k.replace("module.", ""): v for k, v in checkpoint['base_model'].items()}
        classifier.load_state_dict(base_ckpt)
    elif args.method == "PD_LTS_Light" or args.method == "PD_LTS_Heavy":
        checkpoint = torch.load(os.path.join(BASE_DIR, "checkpoints", f"{args.method}.ckpt"), weights_only=True)
        classifier.load_state_dict(checkpoint)
        if args.method == "PD_LTS_Heavy":
            for i in range(3):
                classifier[i].init_as_trained_state()
                classifier[i].eval()
    elif args.method == "PointCleanNet":
        outlier_checkpoint = torch.load(os.path.join(BASE_DIR, "checkpoints", f"{args.method}_Outliers.pth"), weights_only=True)
        denoise_checkpoint = torch.load(os.path.join(BASE_DIR, "checkpoints", f"{args.method}_DeNoise.pth"), weights_only=True)
        classifier_outlier.load_state_dict(outlier_checkpoint)
        classifier_denoise.load_state_dict(denoise_checkpoint)
    else:
        if inspect.isclass(Network):
            checkpoint = torch.load(os.path.join(BASE_DIR, "checkpoints", f"{args.method}.pth"), weights_only=True)
            base_ckpt = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
            classifier.load_state_dict(base_ckpt)
        else:
            pass
    if args.memory:
        if args.method == "PointCleanNet":
            for name, module in classifier_outlier.named_children(): 
                register_hooks(module)
            for name, module in classifier_denoise.named_children(): 
                register_hooks(module)
        else:
            for name, module in classifier.named_children(): 
                register_hooks(module)
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        #with record_function("forward_pass"):
            #output = classifier(xyz)
    '''FORWARD'''
    match args.method:
        case "PcBD" :
            if args.Boundary3D:
                directions = fibonacci_sphere_samples(num=32)
                rotations = [get_rotation_to_align_z(v) for v in directions]
                all_boundary_points = []
                with torch.no_grad():
                    cleaned_xyz = classifier(xyz)[1] 
                    for R in tqdm(rotations, desc='processing:'):
                        R = R.to(xyz.device)
                        rotated = (cleaned_xyz[0] @ R.T).unsqueeze(0)  # (1, N, 3)
                        boundary = classifier(rotated)[-1][0] 
                        boundary = boundary @ R
                        all_boundary_points.append(boundary)
                fused = torch.cat(all_boundary_points, dim=0)
                fused = torch.unique(fused, dim=0)  
                print(fused.size())
                np.savetxt(os.path.join(TEST_PATH, "3D_boundary.txt"), fused.detach().cpu().numpy())
            else:
                output = classifier(xyz)
                _, cleaned_xyz, _, bound_xyz, moved_xyz = output
                print(cleaned_xyz[0].size())
                cleaned_xyz = cleaned_xyz[0].detach().cpu().numpy()
                boundary = bound_xyz[0].detach().cpu().numpy()
                moved_boundary = moved_xyz[0].detach().cpu().numpy()
                np.savetxt(os.path.join(TEST_PATH, "outlier_removed.txt"), cleaned_xyz)
                np.savetxt(os.path.join(TEST_PATH, "boundary.txt"), boundary)
                np.savetxt(os.path.join(TEST_PATH, "smoothed_boundary.txt"), moved_boundary)
                if not args.object.endswith(".pcd"):
                    labels = torch.from_numpy(labels).float().cuda().unsqueeze(0)
                    clabel = labels[:,:,1].unsqueeze(-1)
                    cxyz = score_extract(xyz.permute(0,2,1).contiguous(), clabel.permute(0,2,1), reserve_high=False).permute(0,2,1)
                    blabel = labels[:,:,0].unsqueeze(-1)
                    bxyz = score_extract(cxyz.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
                    np.savetxt(os.path.join(TEST_PATH, "outgt.txt"), cxyz[0].detach().cpu().numpy())
                    np.savetxt(os.path.join(TEST_PATH, "boundgt.txt"), bxyz[0].detach().cpu().numpy())
        case  "ScoreDeNoise" | "DMRDeNoise" :
            output = classifier(xyz)
            cleaned_xyz = output[-1][0].unsqueeze(0)
            print(cleaned_xyz[0].size())
            np.savetxt(os.path.join(TEST_PATH, "denoised.txt"), cleaned_xyz.squeeze().detach().cpu().numpy())
            if not args.object.endswith(".pcd"):
                labels = torch.from_numpy(labels).float().cuda().unsqueeze(0)
                blabel = labels[:,:,0].unsqueeze(-1)
                boundary = score_extract(cleaned_xyz.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
                boundary = boundary.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(TEST_PATH, "smoothed_boundary.txt"), boundary)  
        case "PD_LTS_Light" | "PD_LTS_Heavy":
            xyz = xyz.squeeze()
            if args.method == "PD_LTS_Heavy":
                cleaned_xyz = patch_denoise(classifier, xyz, args.patch_size, 3)
            else:
                cleaned_xyz = light_patch_denoise(classifier, xyz, args.patch_size, 3)
            print(cleaned_xyz.size())
            np.savetxt(os.path.join(TEST_PATH, "denoised.txt"), cleaned_xyz.squeeze().detach().cpu().numpy())
            if not args.object.endswith(".pcd"):
                labels = torch.from_numpy(labels).float().cuda().unsqueeze(0)
                blabel = labels[:,:,0].unsqueeze(-1)
                boundary = score_extract(cleaned_xyz.unsqueeze(0).permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
                boundary = boundary.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(TEST_PATH, "smoothed_boundary.txt"), boundary)
        case "ROR" | "SOR" | "DBSCAN":
            cleaned_xyz = classifier(xyz)
            print(cleaned_xyz.size())
            np.savetxt(os.path.join(TEST_PATH, "outlier_removed.txt"), cleaned_xyz.squeeze().detach().cpu().numpy())
        case "AlphaShapes" | "AdaAlphaShapes" | "GridContour" | "NormalComparing":
            labels = torch.from_numpy(labels).float().cuda().unsqueeze(0)
            clabel = labels[:,:,1].unsqueeze(-1)
            cxyz = score_extract(xyz.permute(0,2,1).contiguous(), clabel.permute(0,2,1), reserve_high=False).permute(0,2,1)
            boundary = classifier(cxyz)
            print(boundary.size())
            np.savetxt(os.path.join(TEST_PATH, "boundary.txt"), boundary.squeeze().detach().cpu().numpy())
        case "BilateralFilter" | "IterNormFilter" | "W_MultiProj" | "MLS" | "AdaMLS" | "SparseReg":
            labels = torch.from_numpy(labels).float().cuda().unsqueeze(0)
            clabel = labels[:,:,1].unsqueeze(-1)
            cxyz = score_extract(xyz.permute(0,2,1).contiguous(), clabel.permute(0,2,1), reserve_high=False).permute(0,2,1)
            denoised_xyz = classifier(cxyz)
            print(denoised_xyz.size())
            np.savetxt(os.path.join(TEST_PATH, "denoised.txt"), denoised_xyz.squeeze().detach().cpu().numpy())
            if not args.object.endswith(".pcd"):
                blabel = labels[:,:,0].unsqueeze(-1)
                boundary = score_extract(denoised_xyz.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
                boundary = boundary.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(TEST_PATH, "smoothed_boundary.txt"), boundary)
        case "PCDNF" | "PointFilter" | "PointCleanNet":
            if args.method == "PointCleanNet":
                _, _, _, xyz_patches = sample_and_group_knn(xyz.permute(0,2,1).contiguous(), points=None, npoint=args.num_points, k=args.patch_size) 
                xyz_patches = xyz_patches.permute(0, 3, 1, 2) #[B, k, 3, N]
                outlier_labels = torch.zeros((1, args.num_points, 1), device=xyz.device) 
                for i in tqdm(range(args.num_points), desc='outlier_removing:'):
                    input_patch = xyz_patches[:, :, :, i].repeat(2,1,1)
                    with torch.no_grad():
                        output = classifier_outlier(input_patch)[0].reshape(1,1)
                        outlier_labels[:,i] = output
                xyz = score_extract(xyz.permute(0,2,1).contiguous(), outlier_labels.permute(0,2,1), threshold=0.5, reserve_high=False).permute(0,2,1)
                np.savetxt(os.path.join(TEST_PATH, "outlier_removed.txt"), xyz.squeeze().detach().cpu().numpy())
                _, _, _, xyz_patches = sample_and_group_knn(xyz.permute(0,2,1).contiguous(), points=None, npoint=args.num_points, k=args.patch_size) 
                xyz_patches = xyz_patches.permute(0, 3, 1, 2) #[B, k, 3, N]
                cleaned_xyz = torch.zeros((1, args.num_points, 3), device=xyz.device)
                for i in tqdm(range(args.num_points), desc='denoising:'):
                    input_patch = xyz_patches[:, :, :, i].repeat(2,1,1)
                    with torch.no_grad():
                        move_vec = classifier_denoise(input_patch)[0].reshape(1,3)
                    new_point = input_patch[0, 0, :].detach().clone().reshape(1,3)
                    new_point += move_vec.detach() * 0.005
                    cleaned_xyz[:, i, :] = new_point 
                print(cleaned_xyz[0].size())
                np.savetxt(os.path.join(TEST_PATH, "denoised.txt"), cleaned_xyz.squeeze().detach().cpu().numpy())
                if not args.object.endswith(".pcd"):
                    labels = torch.from_numpy(labels).float().cuda().unsqueeze(0)
                    blabel = labels[:,:,0].unsqueeze(-1)
                    boundary = score_extract(cleaned_xyz.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
                    boundary = boundary.squeeze().detach().cpu().numpy()
                    np.savetxt(os.path.join(TEST_PATH, "smoothed_boundary.txt"), boundary)
            else:
                labels = torch.from_numpy(labels).float().cuda().unsqueeze(0)
                blabel = labels[:,:,0].unsqueeze(-1)
                bxyz = score_extract(xyz.permute(0,2,1).contiguous(), blabel.permute(0,2,1), reserve_high=True).permute(0,2,1)
                _, _, _, xyz_patches = sample_and_group_knn(bxyz.permute(0,2,1).contiguous(), points=None, npoint=args.num_points, k=args.patch_size) 
                xyz_patches = xyz_patches.permute(0, 3, 1, 2) #[B, k, 3, N]
                cleaned_xyz = torch.zeros((1, args.num_points, 3), device=xyz.device) 
                for i in tqdm(range(args.num_points), desc='denoising:'):
                    input_patch = xyz_patches[:, :, :, i]
                    with torch.no_grad():
                        move_vec = classifier(input_patch)
                        _, noise_inv = pca_alignment(input_patch.squeeze().detach().cpu())
                        noise_inv = torch.from_numpy(noise_inv).float().cuda()
                        move_vec = torch.bmm(noise_inv.unsqueeze(0), move_vec.reshape(1,3,1)).squeeze(-1)
                    new_point = input_patch[:, 0, :].detach().clone()  
                    new_point += move_vec.detach() * 0.005
                    cleaned_xyz[:, i, :] = new_point 
                print(cleaned_xyz[0].size())
                np.savetxt(os.path.join(TEST_PATH, "smoothed_boundary.txt"), cleaned_xyz.squeeze().detach().cpu().numpy())

    
if __name__ == '__main__':

    start = time.perf_counter()
    processed = main()
    end = time.perf_counter()
    print(f"process time: {end - start:.4f} seconds")

    