import os
from tqdm import tqdm
import numpy as np

import h5py
import open3d

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

def pc_normalize(pc, centroid, m):
    pc = pc - centroid
    pc = pc / (m)
    return pc

def batch_normalize(root_dir):
    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    with tqdm(subfolders) as t1:
        for folder in t1:
            model_path = os.path.join(root_dir, folder)
            object_folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
            with tqdm(object_folders) as t2:
                for mobject in t2:
                    object_path = os.path.join(model_path, mobject)
                    obj_folders = [f for f in os.listdir(object_path) if os.path.isdir(os.path.join(object_path, f))]
                
                    for obj in obj_folders:
                        input_path = os.path.join(object_path, obj, "input.pcd")
                        gt_path = os.path.join(object_path, obj, "gt.pcd")
                        pc = IO.get(input_path).astype(np.float32)
                        gt = IO.get(gt_path).astype(np.float32)
                        min_vals = np.min(pc, axis=0)
                        max_vals = np.max(pc, axis=0)
                        ranges = max_vals - min_vals
                        max_size = np.sqrt(np.sum(np.square(ranges)))
                        pc = pc / (max_size)
                        gt = gt / (max_size)
                        ppoint_cloud = open3d.geometry.PointCloud()
                        ppoint_cloud.points = open3d.utility.Vector3dVector(pc)
                        gtpoint_cloud = open3d.geometry.PointCloud()
                        gtpoint_cloud.points = open3d.utility.Vector3dVector(gt)
                        open3d.io.write_point_cloud(input_path, ppoint_cloud)
                        open3d.io.write_point_cloud(gt_path, gtpoint_cloud)

def normalize_point_cloud(point_cloud):
    # 计算x, y, z方向的最大和最小值
    min_vals = np.min(point_cloud, axis=0)
    max_vals = np.max(point_cloud, axis=0)

    # 计算x, y, z方向上的最大距离
    ranges = max_vals - min_vals

    # 计算最大尺寸（x, y, z向最大距离的平方和之平方根）
    max_size = np.sqrt(np.sum(np.square(ranges)))

    # 对点云按最大尺寸归一化
    normalized_point_cloud = point_cloud / max_size

    return normalized_point_cloud
    
if __name__ == '__main__':                   
    root_dir = "/root/PcBD/data/Bound57"  # 将此替换为你的根目录路径
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")
    val_dir = os.path.join(root_dir, "val")
    with tqdm([train_dir, test_dir, val_dir], desc=f'Type') as t1:
        for pdir in [train_dir, test_dir, val_dir]:
            batch_normalize(pdir)