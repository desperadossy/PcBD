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
warnings.filterwarnings("ignore")
import concurrent.futures
import multiprocessing
from provider import sample_edge, cal_bound, cal_bound_union, alpha_shape, noising
from normalize import IO
import cv2

TOF_width = 640
TOF_height = 480
proj_width = 2560
proj_height = 1440
zfar = 5.0

def savepcd(path, cloud):
    pc= o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(cloud)
    o3d.io.write_point_cloud(path, pc)

def duo_pc_normalize(pc, bound):

    coords = pc[:,:3]
    centroid = np.mean(coords, axis=0)
    coords = coords - centroid
    m = np.max(np.sqrt(np.sum(coords**2, axis=1)))
    coords = coords / (2*m)
    pc[:,:3]=coords
    bound = bound - centroid
    bound = bound / (2*m)
    return pc, bound

from scipy.spatial import KDTree
"""
def movevec(noised_cloud, proj_pc):
    # 扩展noised_cloud到8维
    extended_noised_cloud = np.hstack((noised_cloud, np.zeros((noised_cloud.shape[0], 3))))
    tree = KDTree(proj_pc)
    for i, point in enumerate(noised_cloud):
        if point[3] == 1:  # 如果是边界点
            distance, index = tree.query(point[:3])
            nearest_point = proj_pc[index]
            displacement = nearest_point - point[:3]
            extended_noised_cloud[i, 5:] = displacement
    return extended_noised_cloud"""

def gen_depth_map(meshes, cameras, height, width, device, Proj=True):
    
    raster_settings = RasterizationSettings(
        image_size=(height, width), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        #bin_size=1000,
        max_faces_per_bin=1000000,)
    rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings,)
    # create mesh
    
    shader = HardDepthShader(device = device, cameras = cameras) if Proj else HardDepthShader(device = device, cameras = cameras)
    renderer = MeshRenderer(rasterizer, shader)
    images = renderer(meshes)
    depth_map = images.squeeze().cpu().numpy()
    return depth_map


def pc_projection(cameras, depth_map, device, zfar=5.0):
    height, width = depth_map.shape
    y, x = np.mgrid[0:height, 0:width]
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    camera_x = torch.tensor((width / 2 - x) / (height / 2), device=device)
    camera_y = torch.tensor((height / 2 - y) / (height / 2), device=device)
    camera_z = torch.tensor(depth_map, device=device)
    world_coords = torch.stack([camera_x, camera_y, camera_z], dim=-1)
    world_coords = cameras.unproject_points(world_coords, world_coordinates=False)  
    world_coords_np = world_coords.cpu().numpy().reshape(-1, 3)
    world_coords_np = world_coords_np[np.floor(world_coords_np[:, 2]) != zfar]
    return world_coords_np


def back_projection(cameras, depth_map, contours, device, zfar=5.0, pc=True):
    height, width = depth_map.shape
    y, x = np.mgrid[0:height, 0:width]
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    camera_x = torch.tensor((width / 2 - x) / (height / 2), device=device)
    camera_y = torch.tensor((height / 2 - y) / (height / 2), device=device)
    camera_z = torch.tensor(depth_map, device=device)

    """
    #光学失真
    k1=0.01
    k2=0.001
    r = torch.sqrt(camera_x**2 + camera_y**2)  # 径向距离
    scale = 1 + k1 * r**2 + k2 * r**4  # 畸变因子
    camera_x_distorted = camera_x * scale
    camera_y_distorted = camera_y * scale

    p1=0.001
    p2=0.001
    x_tangential = camera_x_distorted + (2 * p1 * camera_x * camera_y + p2 * (r**2 + 2 * camera_x**2))
    y_tangential = camera_y_distorted + (p1 * (r**2 + 2 * camera_y**2) + 2 * p2 * camera_x * camera_y)

    camera_x = x_tangential
    camera_y = y_tangential
    """


    world_coords = torch.stack([camera_x, camera_y, camera_z], dim=-1)
    world_coords = cameras.unproject_points(world_coords, world_coordinates=False)  
    world_coords_np = world_coords.cpu().numpy().reshape(height, width, 3)
    if pc:
        world_coords_np_4d = np.zeros((height, width, 4))
        world_coords_np_4d[:, :, :3] = world_coords_np
    else:
        bounds = set()
    for contour in contours:
        for point in contour:
            x, y = point[0]
            if depth_map[y, x] == zfar:
                # 搜索周围邻域的有效点
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1), (-1, -1), (1, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and depth_map[ny, nx] != zfar:
                        count = 0
                        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            nnx, nny = nx + ddx, ny + ddy
                            if 0 <= nnx < width and 0 <= nny < height and depth_map[nny, nnx] != zfar:
                                count += 1
                        if count < 4:
                            if pc:
                                world_coords_np_4d[ny, nx, 3] = 1
                            else:
                                bounds.add(tuple(world_coords_np[ny, nx]))
            else:
                if pc:
                    world_coords_np_4d[y, x, 3] = 1
                else:
                    bounds.add(tuple(world_coords_np[y, x]))
    if pc:
        world_coords_np_4d = world_coords_np_4d.reshape(-1, 4)
        world_coords_np_4d = world_coords_np_4d[np.floor(world_coords_np_4d[:, 2]) != zfar]
        return world_coords_np_4d
    else:
        bounds = np.array(list(bounds)).reshape(-1, 3)
        bounds = bounds[np.floor(bounds[:, 2]) != zfar]
        return bounds

def depth_to_contour(depth_map, name):
    background_color = depth_map[0, 0]
    binary_thresh = np.where(depth_map == background_color, 0, 255).astype(np.uint8)
    #cv2.imwrite('binary_thresh.png', binary_thresh)
    edges = cv2.Canny(binary_thresh, 0, 255)
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    vis = cv2.drawContours(np.zeros_like(depth_map), contours, -1, (255, 255, 255), 1)
    #cv2.imwrite(f'{name}_contour.png', vis)
    return contours


