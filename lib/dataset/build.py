# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data

from .COCODataset import CocoDataset as coco
from .COCOKeypoints import CocoKeypoints as coco_kpt
from .CrowdPoseDataset import CrowdPoseDataset as crowd_pose
from .CrowdPoseKeypoints import CrowdPoseKeypoints as crowd_pose_kpt
from .transforms import build_transforms
from .target_generators import HeatmapGenerator
from .target_generators import ScaleAwareHeatmapGenerator
from .target_generators import JointsGenerator
from .HIEKeypoints import HIEKeypoints as hie_kpt
## 
import os
from os.path import join as opj
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

def build_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)

    if cfg.DATASET.SCALE_AWARE_SIGMA:
        # _C.DATASET.SCALE_AWARE_SIGMA = False
        _HeatmapGenerator = ScaleAwareHeatmapGenerator
    else:
        _HeatmapGenerator = HeatmapGenerator

    heatmap_generator = [
        _HeatmapGenerator(
            output_size, 
            cfg.DATASET.NUM_JOINTS,
            cfg.DATASET.SIGMA
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]       # DATASET.SIGMA = -1, DATASET.NUM_JOINTS = 17
            # DATASET.OUTPUT_SIZE = [128, 256, 512]
    joints_generator = [
        JointsGenerator(
            cfg.DATASET.MAX_NUM_PEOPLE,
            cfg.DATASET.NUM_JOINTS,
            output_size,
            cfg.MODEL.TAG_PER_JOINT
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]
            # DATASET.MAX_NUM_PEOPLE = 30
            # DATASET.NUM_JOINTS = 17
            # MODEL.TAG_PER_JOINT = True
    # dataset_name = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST
            # DATASET.TRAIN = 'train2017'
    dataset_name = 'train'
    # DATASET.DATASET = 'coco_kpt'
    dataset = eval('hie_kpt')(
        cfg,
        dataset_name,
        is_train,
        heatmap_generator,
        joints_generator,
        transforms
    )
    # eval('coco_kpt') => CocoKeypoints
    # (self,cfg,dataset_name,remove_images_without_annotations,
    # heatmap_generator,joints_generator,transforms=None)
    return dataset


def make_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        # _C.TRAIN.IMAGES_PER_GPU = 32
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    images_per_batch = images_per_gpu * len(cfg.GPUS)

    dataset = build_dataset(cfg, is_train) 

    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset
        )
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler
    )

    return data_loader


def make_test_dataloader(cfg):
    transforms = None
    dataset = eval(cfg.DATASET.DATASET_TEST)(
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST,
        cfg.DATASET.DATA_FORMAT,
        transforms
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset

class HIE_dataloader(Dataset):
    def __init__(self,data_path):
        super(HIE_dataloader,self).__init__()
        self.transform = None
        self.data_path = data_path     
        self.image = []
        self.get_images()
    
    def get_images(self):
        folders = os.listdir(self.data_path)
        for folder in folders:
            folder_path = opj(self.data_path,folder)
            images = os.listdir(folder_path)
            for image in images:
                pic_path = opj(self.data_path,folder,image)
                self.image.append(pic_path)

    def __getitem__(self, index):
        
        dataset = np.array(Image.open(self.image[index]))

        return dataset 

    def __len__(self):

        return len(self.image)
    


