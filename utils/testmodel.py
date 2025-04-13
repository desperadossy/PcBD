import numpy as np
import os
import torch
import sys
import importlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

import h5py
import numpy as np
import open3d
import os

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd', '.ply']:
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
    
def group_random_sample(xyz, labels, n_points):
    choice = np.random.permutation(xyz.shape[0])
    xyz = xyz[choice[:n_points]]
    labels = labels[choice[:n_points]]
    if xyz.shape[0] < n_points:
        pzeros = np.zeros((n_points - xyz.shape[0], 3))
        lzeros = np.zeros((n_points - xyz.shape[0], 3))
        xyz = np.concatenate([xyz, pzeros])
        labels = np.concatenate((labels, lzeros))
    return xyz, labels

def random_sample(xyz, n_points):
    choice = np.random.permutation(xyz.shape[0])
    xyz = xyz[choice[:n_points]]
    if xyz.shape[0] < n_points:
        pzeros = np.zeros((n_points - xyz.shape[0], 3))
        xyz = np.concatenate([xyz, pzeros])
    return xyz

def upsample(xyz, n_points):
    curr = xyz.shape[0]
    need = n_points - curr
    if need < 0:
        return xyz[np.random.permutation(n_points)]
    while curr <= need:
        xyz = np.tile(xyz, (2, 1))
        need -= curr
        curr *= 2
    choice = np.random.permutation(need)
    xyz = np.concatenate((xyz, xyz[choice]))
    return xyz

def group_up_sample(xyz, labels, n_points):
    choice = np.random.permutation(xyz.shape[0])
    xyz = xyz[choice[:n_points]]
    labels = labels[choice[:n_points]]
    if xyz.shape[0] < n_points:
        curr = xyz.shape[0]
        need = n_points - curr
        while curr <= need:
            xyz = np.tile(xyz, (2, 1))
            labels = np.tile(labels, (2, 1))
            need -= curr
            curr *= 2
        choice = np.random.permutation(need)
        xyz = np.concatenate([xyz, xyz[choice]])
        labels = np.concatenate((labels, labels[choice]))
    return xyz, labels

def score_extract(points, scores, threshold=0.1, reserve_high=False):
    _, N = points.shape
    if reserve_high:
        high_score_mask = scores > threshold
    else:
        high_score_mask = scores < threshold
    #print(torch.max(scores))
    points = points * high_score_mask
    return points

def main():
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    '''MODEL LOADING'''
    xyz = IO.get('input.pcd').astype(np.float32)  
    labels = IO.get('label.npy').astype(np.float32)  
    xyz, labels = group_up_sample(xyz, labels, 4096)
    np.savetxt('xyz.txt', xyz)
    nlabel = labels[:,1].reshape(-1,1)
    blabel = labels[:,0].reshape(-1,1)  
    print(xyz.shape)
    print(nlabel.shape)
    cxyz = score_extract(np.transpose(xyz,(1,0)), np.transpose(nlabel,(1,0)), reserve_high=False)
    bxyz = score_extract(cxyz, np.transpose(blabel,(1,0)), reserve_high=True)
    np.savetxt('cleaned.txt',  np.transpose(cxyz,(1,0)))
    np.savetxt('bound.txt',  np.transpose(bxyz,(1,0)))
if __name__ == '__main__':
    main()