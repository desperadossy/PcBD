import open3d as o3d
import numpy as np
import torch
from sys import path
import alphashape
import shapely

def polar_angle(point):
    # 计算点的极角
    x, y, _ = point
    return np.arctan2(y, x)

def square_distance(src, dst):

    dist = -2 * np.matmul(src, dst.transpose(1, 0))
    dist += np.sum(src ** 2, -1).reshape(-1, 1)
    dist += np.sum(dst ** 2, -1).reshape(1, -1)
    return dist

def check_probability(prob):
    rand_num = np.random.random()  # 生成0到1之间的随机数
    return rand_num < prob



def up_sample(xyz, n_points):

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

def divide_point_cloud(point_cloud, n=8, flat_distances=None):

    angles = np.array([polar_angle(point) for point in point_cloud[:,:3]])
    sorted_indices = np.argsort(angles)
    sorted_point_cloud = point_cloud[sorted_indices]
    divided_clouds = np.array_split(sorted_point_cloud, n)
    if flat_distances is not None:
        sorted_distances = flat_distances[sorted_indices]
        divided_distances = np.array_split(sorted_distances, n)
        return divided_clouds, divided_distances
    else:
        return divided_clouds
    
def sample_edge(ori_pcd):
    
    pcd = ori_pcd[:,:3]
    flat_pcd = ori_pcd[:,:2]
    nbrs = NearestNeighbors(n_neighbors=32, algorithm='auto').fit(flat_pcd)
    flat_distances, _ = nbrs.kneighbors(flat_pcd)
    avg_flat_distance = np.mean(flat_distances[:, 1:31])
    scaler = pcd.shape[0]/6000
    furthest_points = []
    sparse_points = []
    index = []
    divided_clouds, divided_flat_distances = divide_point_cloud(ori_pcd, np.random.randint(16, 32), flat_distances)
    global_best_judge = 0.0
    for point_patch, flat_distances_patch in zip(divided_clouds, divided_flat_distances):
        coord = point_patch[:,:3]
        best_judge = 0.0
        sparse_judge = 0.0
        bounds = coord[point_patch[:,-1]==1]
        bounds_flat_distances = flat_distances_patch[point_patch[:,-1]==1]
        for b_point, b_flat_distances in zip(bounds, bounds_flat_distances):
            dist = square_distance(pcd, b_point.reshape(1,3)).reshape(1,-1)
            dist_idx = np.argsort(dist).reshape(-1,1)
            if np.mean(b_flat_distances) <= avg_flat_distance:
                judge = np.mean(pcd[dist_idx[1:32]].squeeze()[:, 2] <= b_point[2])
                if(judge>global_best_judge):
                    global_best_judge = judge
                if(judge>best_judge):
                    furthest_points.append((b_point, dist_idx))
                    if (best_judge>0):
                        furthest_points.pop(-2)
                    best_judge = judge       
            elif np.mean(b_flat_distances) > 1.5 * avg_flat_distance:
                if(np.mean(b_flat_distances)>sparse_judge):
                    sparse_points.append((b_point, dist_idx))
                    if (sparse_judge>0):
                        sparse_points.pop(-2)
                    sparse_judge = np.mean(b_flat_distances)    
    for seed, dist_idx in furthest_points:
        furthest_region = np.random.randint(64*scaler, 256*scaler)
        ratio = np.random.randint(2,5)
        out_points_idx = np.random.permutation(dist_idx[1:furthest_region])[(ratio-1) * int(furthest_region/ratio):]
        index.append(out_points_idx)
    for sparse_seed, dist_idx in sparse_points:
        furthest_region = np.random.randint(64*scaler, 128*scaler)
        out_points_idx = np.random.permutation(dist_idx[1:furthest_region])[int(furthest_region/np.random.randint(3,8)):]
        index.append(out_points_idx)

    points = np.array([b_point for b_point, _ in furthest_points])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    #o3d.io.write_point_cloud("seed.ply", pcd)

    points = np.array([b_point for b_point, _ in sparse_points])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    #o3d.io.write_point_cloud("sparse_seed.ply", pcd)

    if(len(index)>0):
        cut_idx = np.vstack(index)
        ori_pcd = np.delete(ori_pcd, cut_idx, 0)
    
    return ori_pcd


def cal_bound(pcd, alpha):
    alpha_shape = alphashape.alphashape(pcd[:,:2], alpha)
    bounds = []
    if isinstance (alpha_shape, shapely.geometry.polygon.Polygon):
        boundary_points = np.array(alpha_shape.exterior.coords.xy).T
        mask = np.any((pcd[:,:2][:, None, :] == boundary_points).all(axis=2), axis=1)
        output = np.zeros((pcd.shape[0], 4))
        output[:, :3] = pcd
        output[mask, 3] = 1
        return output
    else:
        for polygon in alpha_shape.geoms:
            boundary_points = np.array(polygon.exterior.coords.xy).T
            bounds.append(boundary_points)
            bounds = np.vstack(bounds)
            mask = np.any((pcd[:,:2][:, None, :] == bounds).all(axis=2), axis=1)
            output = np.zeros((pcd.shape[0], 4))
            output[:, :3] = pcd
            output[mask, 3] = 1
            return output

def cal_bound_union(pcd, alphas):
    all_bounds = set()  # Use a set to store unique boundary points

    for alpha in alphas:
        alpha_shape = alphashape.alphashape(pcd[:, :2], alpha)
        
        if isinstance(alpha_shape, shapely.geometry.polygon.Polygon):
            boundary_points = set(map(tuple, np.array(alpha_shape.exterior.coords.xy).T))
            all_bounds.update(boundary_points)
        else:
            for polygon in alpha_shape.geoms:
                boundary_points = set(map(tuple, np.array(polygon.exterior.coords.xy).T))
                all_bounds.update(boundary_points)

    # Convert the set of boundary points back to a numpy array
    bounds = np.array(list(all_bounds))
    mask = np.any((pcd[:, :2][:, None, :] == bounds).all(axis=2), axis=1)
    output = np.zeros((pcd.shape[0], 4))
    output[:, :3] = pcd
    output[mask, 3] = 1
    return output


from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def alpha_shape(points, alpha):
    """
    计算给定点集的Alpha Shape（近似）。
    :param points: 二维点集。
    :param alpha: Alpha值，控制边界的紧密程度。
    :return: Alpha Shape的边界线段。
    """
    def add_edge(edges, edge_points, coords, i, j):
        """
        添加边界线段，如果长度小于指定的Alpha值。
        """
        if (i, j) in edges or (j, i) in edges:
            # 如果已经添加了这条边，则跳过
            return
        p1 = coords[i]
        p2 = coords[j]
        if np.linalg.norm(p1 - p2) < alpha:
            edges.add((i, j))
            edge_points.append(p1)
            edge_points.append(p2)

    coords = points
    tri = Delaunay(coords)
    edges = set()
    edge_points = []

    # 循环遍历Delaunay三角剖分的三角形，添加边缘
    for ia, ib, ic in tri.simplices:
        add_edge(edges, edge_points, coords, ia, ib)
        add_edge(edges, edge_points, coords, ib, ic)
        add_edge(edges, edge_points, coords, ic, ia)

    return np.array(edge_points)


from sklearn.neighbors import NearestNeighbors
def noising(point_cloud, noise_ratio=np.random.uniform(0.05, 0.1), noise_level_ratio=0.15, xy_noise_multiplier=np.random.uniform(2, 20), z_noise_multiplier=np.random.uniform(50, 100)):
    n = point_cloud.shape[0]  # 原始点云中的点数
    
    # 使用 K 近邻算法找到每个点的最近邻点
    nbrs = NearestNeighbors(n_neighbors=32, algorithm='auto').fit(point_cloud[:, :3])
    distances, indices = nbrs.kneighbors(point_cloud[:, :3])

    # 计算平均距离
    avg_distance = np.mean(distances[:, 1:31])

    # 根据平均距离和噪声比例系数计算实际噪声水平
    noise_level = avg_distance * noise_level_ratio

    m = int(n * noise_ratio)  # 需要添加的噪声点数量
    noisy_point_cloud = point_cloud.copy()
    #光学失真
    k1=0.01
    k2=0.001
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    r = np.sqrt(x**2 + y**2)  # 径向距离
    scale = 1 + k1 * r**2 + k2 * r**4  # 畸变因子
    x_distorted = x * scale
    y_distorted = y * scale

    p1=0.001
    p2=0.001
    x_tangential = x_distorted + (2 * p1 * x * y + p2 * (r**2 + 2 * x**2))
    y_tangential = y_distorted + (p1 * (r**2 + 2 * y**2) + 2 * p2 * x * y)

    noisy_point_cloud[:, 0] = x_tangential
    noisy_point_cloud[:, 1] = y_tangential
    

    # 为原始点生成空间噪声
    spatial_noise = np.random.normal(0, (point_cloud[:, 2]*noise_level).reshape(-1,1), size=(n, 3))
    noisy_point_cloud[:, :3] += spatial_noise

    # 随机选择 m 个点作为噪声点的基准位置
    base_points_indices = np.random.choice(n, m, replace=False)
    base_points = point_cloud[base_points_indices, :3]

    # 生成额外的纯噪声点（Z 轴噪声更大）
    pure_noise_x = np.random.normal(0, noise_level * xy_noise_multiplier, m)
    pure_noise_y = np.random.normal(0, noise_level * xy_noise_multiplier, m)
    pure_noise_z = np.random.normal(0, noise_level * z_noise_multiplier, m)
    pure_noise_points = np.column_stack((base_points[:, 0] + pure_noise_x, 
                                         base_points[:, 1] + pure_noise_y, 
                                         base_points[:, 2] + pure_noise_z))

    noise_boundary_indicator = np.zeros((m, 1))
    pure_noise_indicator = np.ones((m, 1))

    # 合并纯噪声点及其指示器
    additional_noise_points = np.hstack((pure_noise_points, noise_boundary_indicator, pure_noise_indicator))

    # 为原始噪声点云添加纯噪声指示器列（全零）
    final_point_cloud = np.hstack((noisy_point_cloud, np.zeros((n, 1))))

    # 将原始噪声点云与额外的纯噪声点合并
    final_point_cloud = np.vstack((final_point_cloud, additional_noise_points))

    return noisy_point_cloud, final_point_cloud

if __name__ == '__main__':
    data = IO.get('05.pcd').astype(np.float32)
    data = up_sample(data, 2048)
    data = sample_edge(data)
    np.savetxt("gt.txt", data)
