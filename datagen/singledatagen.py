import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, SfMOrthographicCameras, OpenGLOrthographicCameras, FoVPerspectiveCameras, FoVOrthographicCameras, SoftPhongShader, FoVPerspectiveCameras, FoVOrthographicCameras, TexturesAtlas, RasterizationSettings, SoftGouraudShader, MeshRenderer, MeshRasterizer, compositing
from pytorch3d.renderer import HardDepthShader, SoftDepthShader, HardGouraudShader
import os
import open3d as o3d
import json
from tqdm import tqdm
import random
import warnings
#warnings.filterwarnings("ignore")
import concurrent.futures
import multiprocessing
from utils import duo_pc_normalize, savepcd, gen_depth_map, depth_to_contour, back_projection
from provider import sample_edge, noising

def obj_sampling(path_obj, target_folder_path, args):
    
    #uniform-sample obj into ground-truths
    mesh = o3d.io.read_triangle_mesh(path_obj)
    complete_point_cloud = mesh.sample_points_uniformly(number_of_points=args['complete_points'])
    # 保存点云为 PCD 文件
    o3d.io.write_point_cloud(os.path.join(target_folder_path, 'complete.pcd'), complete_point_cloud)


def obj_projection(object_file_path, target_folder_path, args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vertices, faces, aux = load_obj(object_file_path, load_textures=False)
    R, T = look_at_view_transform(np.random.uniform(args['camera_dist_min'], args['camera_dist_max']),
                                np.random.uniform(-90, 90), np.random.uniform(0, 360))
    TOFcamera = FoVPerspectiveCameras(device=device, R=R, T=T, zfar=args['camera_zfar'])
    projcamera = FoVPerspectiveCameras(device=device, R=R, T=T, zfar=args['camera_zfar'])
    mesh = load_objs_as_meshes([object_file_path], device=device)
    TOF_depth = gen_depth_map(mesh, TOFcamera, args['TOF_image_height'], args['TOF_image_width'], device, Proj=False)
    height, width = TOF_depth.shape
    proj_depth = gen_depth_map(mesh, projcamera, args['Proj_image_height'], args['Proj_image_width'], device)
    TOF_contours = depth_to_contour(TOF_depth)
    proj_contours = depth_to_contour(proj_depth)
    TOF_pc = back_projection(TOFcamera, TOF_depth, TOF_contours, device, zfar=args['camera_zfar'])
    proj_pc = back_projection(projcamera, proj_depth, proj_contours, device, zfar=args['camera_zfar'], pc=False)
    TOF_pc = noising(sample_edge(TOF_pc))
    TOF_pc, proj_pc = duo_pc_normalize(TOF_pc, proj_pc)
    savepcd(os.path.join(target_folder_path, "input.pcd"), TOF_pc[:, :3])
    np.save(os.path.join(target_folder_path, "label.npy"), TOF_pc[:, 3:5])
    savepcd(os.path.join(target_folder_path, "gt.pcd"), proj_pc)


def process_object(source_object_path, target_object_path, args):
    object_file_path = os.path.join(source_object_path, "model.obj")
    if os.path.exists(object_file_path):
        for i in range(args['view_num']):
            split_target_folder = os.path.join(target_object_path, f"{i:02d}")
            os.makedirs(split_target_folder, exist_ok=True)
            obj_projection(object_file_path, split_target_folder, args)

def process_single_object(source_object_path, target_object_path, args):
    object_file_path = os.path.join(source_object_path, "model.obj")
    if os.path.exists(object_file_path):
        split_target_folder = os.path.join(target_object_path, "00")
        os.makedirs(split_target_folder, exist_ok=True)
        obj_projection(object_file_path, split_target_folder, args)


def process_folder(objects, category_path, target_folder, category_folder, args):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for object_folder in objects:
            source_object_path = os.path.join(category_path, object_folder)
            target_object_path = os.path.join(target_folder, category_folder, object_folder)
            os.makedirs(target_object_path, exist_ok=True)
            futures.append(
                executor.submit(
                    process_object,
                    source_object_path,
                    target_object_path,
                    args,
                )
            )

        # Wait for all threads to finish
        concurrent.futures.wait(futures)

def process_single_folder(objects, category_path, target_folder, category_folder, args):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for object_folder in objects:
            source_object_path = os.path.join(category_path, object_folder)
            target_object_path = os.path.join(target_folder, category_folder, object_folder)
            os.makedirs(target_object_path, exist_ok=True)
            futures.append(
                executor.submit(
                    process_single_object,
                    source_object_path,
                    target_object_path,
                    args,
                )
            )

        # Wait for all threads to finish
        concurrent.futures.wait(futures)

def gen_data(args):
    # Create train, val, test subdirectories
    train_folder = os.path.join(args['target_dir'], 'train')
    val_folder = os.path.join(args['target_dir'], 'val')
    test_folder = os.path.join(args['target_dir'], 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    category_subfolders = [f for f in os.listdir(args['source_dir']) if os.path.isdir(os.path.join(args['source_dir'], f))]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        with tqdm(category_subfolders) as t1:
            for category_folder in t1:
                category_path = os.path.join(args['source_dir'], category_folder)
                object_subfolders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
                total_objects = len(object_subfolders)
                train_count = int(0.8 * total_objects)
                val_count = int(0.1 * total_objects)
                train_objects = random.sample(object_subfolders, train_count)
                remaining_objects = list(set(object_subfolders) - set(train_objects))
                val_objects = random.sample(remaining_objects, val_count)
                test_objects = list(set(remaining_objects) - set(val_objects))           
                futures.append(
                    executor.submit(
                        process_folder,
                        train_objects,
                        category_path,
                        train_folder,
                        category_folder,
                        args,
                    )
                )
                futures.append(
                    executor.submit(
                        process_folder,
                        val_objects,
                        category_path,
                        val_folder,
                        category_folder,
                        args,
                    )
                )
                futures.append(
                    executor.submit(
                        process_single_folder,
                        test_objects,
                        category_path,
                        test_folder,
                        category_folder,
                        args,
                    )
                )

        # Wait for all threads to finish
        concurrent.futures.wait(futures)

if __name__ == '__main__':
    object_file_path = "/root/PcBD/datagen/example/model.obj"
    target_folder_path = "/root/PcBD/datagen/example"
    zfar = 5.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    R, T = look_at_view_transform(np.random.uniform(1.25, 3.0), np.random.uniform(-90, 90), np.random.uniform(0, 360))
    #R, T = look_at_view_transform(2.0, 0.0, 0.0)
    TOFcamera = FoVPerspectiveCameras(device=device, R=R, T=T, zfar=zfar)
    projcamera = FoVPerspectiveCameras(device=device, R=R, T=T, zfar=zfar)
    mesh = load_objs_as_meshes([object_file_path], device=device)

    TOF_depth = gen_depth_map(mesh, TOFcamera, 480, 640, device, Proj=False)
    plt.imsave(os.path.join(target_folder_path, "TOF_depth.png"), TOF_depth, cmap='gray')

    proj_depth = gen_depth_map(mesh, projcamera, 1440, 2560, device)
    plt.imsave(os.path.join(target_folder_path, "proj_depth.png"), proj_depth, cmap='gray')

    TOF_contours = depth_to_contour(TOF_depth, "TOF")
    proj_contours = depth_to_contour(proj_depth, "proj")

    TOF_pc_raw = back_projection(TOFcamera, TOF_depth, TOF_contours, device, zfar)
    savepcd(os.path.join(target_folder_path, "TOF_pc_raw.pcd"), TOF_pc_raw[:, :3])

    # Projected boundary ground truth
    proj_pc = back_projection(projcamera, proj_depth, proj_contours, device, zfar, pc=False)
    savepcd(os.path.join(target_folder_path, "proj_pc.pcd"), proj_pc[:, :3])

    # For upsampling or reference (dense ground truth)
    ups_pc = back_projection(projcamera, proj_depth, proj_contours, device, zfar, pc=True)
    savepcd(os.path.join(target_folder_path, "ups_pc.pcd"), ups_pc[:, :3])

    # Sample edge and add noise
    sampled_pc = sample_edge(TOF_pc_raw)
    savepcd(os.path.join(target_folder_path, "sampled_pc.pcd"), sampled_pc[:, :3])

    only_noised, TOF_pc = noising(sampled_pc)
    savepcd(os.path.join(target_folder_path, "only_noised.pcd"), only_noised[:, :3])
    savepcd(os.path.join(target_folder_path, "TOF_pc_noised.pcd"), TOF_pc[:, :3])

    # Generate per-point labels (TOF_pc shape: [N, 5], label in [:, 3:5])
    np.save(os.path.join(target_folder_path, "label.npy"), TOF_pc[:, 3:5])

    # Normalize all results together
    TOF_pc, proj_pc, ups_pc, only_noised = multi_pc_normalize(TOF_pc, proj_pc, ups_pc, only_noised)

    # Final outputs for standard pipeline
    savepcd(os.path.join(target_folder_path, "input.pcd"), TOF_pc[:, :3])
    savepcd(os.path.join(target_folder_path, "gt.pcd"), proj_pc)




