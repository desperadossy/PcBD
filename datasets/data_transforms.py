# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-02 14:38:36
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-03 09:23:07
# @Email:  cshzxie@gmail.com

import numpy as np
import torch
import transforms3d

class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable
    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)
            if transform.__class__ in [NormalizeObjectPose, GroupRandomSamplePoints, GroupUpSamplePoints, GroupShuffle]:
                data = transform(data)
            else:
                for k, v in data.items():
                    if k in objects and k in data:
                        if transform.__class__ in [
                             RandomMirrorPoints
                        ]:
                            data[k] = transform(v, rnd_value)
                        else:
                            data[k] = transform(v)

        return data

class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud


class GroupRandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, data):
        ptcloud = data['input_cloud']
        labels = data['labels']
        #mv = data['move_vec']
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]
        labels = labels[choice[:self.n_points]]
        #mv = mv[choice[:self.n_points]]
        if ptcloud.shape[0] < self.n_points:
            pzeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            lzeros = np.zeros((self.n_points - ptcloud.shape[0], 2))
            ptcloud = np.concatenate([ptcloud, pzeros])
            labels = np.concatenate((labels, lzeros))
        data['input_cloud'] = ptcloud
        data['labels'] = labels
        #data['move_vec'] = mv
        return data


class GroupUpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, data):
        ptcloud = data['input_cloud']
        labels = data['labels']
        #mv = data['move_vec']
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]
        labels = labels[choice[:self.n_points]]
        #mv = mv[choice[:self.n_points]]
        if ptcloud.shape[0] < self.n_points:
            curr = ptcloud.shape[0]
            need = self.n_points - curr
            while curr <= need:
                ptcloud = np.tile(ptcloud, (2, 1))
                labels = np.tile(labels, (2, 1))
                need -= curr
                curr *= 2
            choice = np.random.permutation(need)
            ptcloud = np.concatenate((ptcloud, ptcloud[choice]))
            labels = np.concatenate((labels, labels[choice]))
        data['input_cloud'] = ptcloud
        data['labels'] = labels
        #data['move_vec'] = mv
        return data
    
    
class UpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr
        if need < 0:
            return ptcloud[np.random.permutation(self.n_points)]
        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2
        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud

class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value > 0.5 and rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]

        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data


class GroupShuffle(object):
    def __init__(self, parameters):
        pass

    def __call__(self, data):
        ptcloud = data['input_cloud']
        labels = data['labels']
        #mv = data['move_vec']
        choice = np.random.permutation(ptcloud.shape[0])
        indices = np.arange(ptcloud.shape[0])
        np.random.shuffle(indices)
        ptcloud = ptcloud[indices]
        labels = labels[indices]
        #mv = mv[indices]
        data['input_cloud'] = ptcloud
        data['labels'] = labels
        #data['move_vec'] = mv
        return data
