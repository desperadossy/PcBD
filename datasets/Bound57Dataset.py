import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *

# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class Bound57(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.input_points_path = config.INPUT_POINTS_PATH
        self.labels_path = config.LABEL_PATH
        self.gt_path = config.GT_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        
        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)
        


    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'GroupUpSamplePoints',
                'parameters': {
                    'n_points': self.npoints
                },
                'objects': ['input_cloud', 'labels']
                }, {
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': self.npoints
                },
                'objects': ['gt']
                }, {
                'callback': 'ToTensor',
                'objects': ['input_cloud', 'labels', 'gt']
             }])
        else:
            return data_transforms.Compose([{
                'callback': 'GroupUpSamplePoints',
                'parameters': {
                    'n_points': self.npoints
                },
                'objects': ['input_cloud', 'labels']
                }, {
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': self.npoints
                },
                'objects': ['gt']
                }, {
                'callback': 'ToTensor',
                'objects': ['input_cloud', 'labels', 'gt']
             }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='BOUND57DATASET')
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'input_cloud_path': [
                        self.input_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'labels_path': [
                        self.labels_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path': [
                        self.gt_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='BOUND57DATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['input_cloud', 'labels', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['input_cloud'], data['labels'], data['gt'])

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class Bound57SingleCategory(data.Dataset):
    def __init__(self, config):
        self.input_points_path = config.INPUT_POINTS_PATH
        self.labels_path = config.LABEL_PATH
        self.gt_path = config.GT_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset

        # Load and filter dataset categories to only include '02691156'
        with open(self.category_file) as f:
            all_categories = json.loads(f.read())
        self.dataset_categories = [
            dc for dc in all_categories if dc['taxonomy_id'] == '02691156'
        ]

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        return data_transforms.Compose([
            {
                'callback': 'GroupUpSamplePoints',
                'parameters': {'n_points': self.npoints},
                'objects': ['input_cloud', 'labels']
            },
            {
                'callback': 'UpSamplePoints',
                'parameters': {'n_points': self.npoints},
                'objects': ['gt']
            },
            {
                'callback': 'ToTensor',
                'objects': ['input_cloud', 'labels', 'gt']
            }
        ])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset — only for taxonomy_id='02691156'"""
        file_list = []
        taxonomy_id = '02691156'  # 固定分类 ID

        for dc in self.dataset_categories:
            if dc['taxonomy_id'] != taxonomy_id:
                continue  # 跳过非指定分类

            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='BOUND57SINGLE')
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id': taxonomy_id,
                    'model_id': s,
                    'input_cloud_path': [
                        self.input_points_path % (subset, s, i)
                        for i in range(n_renderings)
                   ],
                    'labels_path': [
                        self.labels_path % (subset, s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path': [
                        self.gt_path % (subset, s, i)
                        for i in range(n_renderings)
                    ],
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='BOUND57SINGLE')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset == 'train' else 0

        for ri in ['input_cloud', 'labels', 'gt']:
            file_path = sample[f'{ri}_path']
            if isinstance(file_path, list):
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['input_cloud'], data['labels'], data['gt'])

    def __len__(self):
        return len(self.file_list)