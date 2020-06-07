# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import transforms as T


FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'COCO_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ],
    'CROWDPOSE': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13
    ],
    'CROWDPOSE_WITH_CENTER': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14
    ],
    'HIE20':[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
    ]   # wait for change
}


def build_transforms(cfg, is_train=True):
    assert is_train is True, 'Please only use build_transforms for training.'
    assert isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)), 'DATASET.OUTPUT_SIZE should be list or tuple'
    if is_train:
        max_rotation = cfg.DATASET.MAX_ROTATION     #30
        min_scale = cfg.DATASET.MIN_SCALE           # 0.75
        max_scale = cfg.DATASET.MAX_SCALE           # 1.25
        max_translate = cfg.DATASET.MAX_TRANSLATE   # 40
        input_size = cfg.DATASET.INPUT_SIZE         # 512
        output_size = cfg.DATASET.OUTPUT_SIZE       # [128, 256, 512]
        flip = cfg.DATASET.FLIP                     # 0.5
        scale_type = cfg.DATASET.SCALE_TYPE         # 'short'
    else:
        scale_type = cfg.DATASET.SCALE_TYPE
        max_rotation = 0
        min_scale = 1
        max_scale = 1
        max_translate = 0
        input_size = 512
        output_size = [128]
        flip = 0

    # coco_flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    # if cfg.DATASET.WITH_CENTER:
        # coco_flip_index.append(17)
    if 'coco' in cfg.DATASET.DATASET:
        dataset_name = 'COCO'
    elif 'crowd_pose' in cfg.DATASET.DATASET:
        dataset_name = 'CROWDPOSE'
    elif True:
        dataset_name = 'HIE20'
    else:
        raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
    if cfg.DATASET.WITH_CENTER:
        # DATASET.WITH_CENTER = False
        coco_flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER']
    else:
        coco_flip_index = FLIP_CONFIG[dataset_name]

    transforms = T.Compose(
        [
            T.RandomAffineTransform(
                input_size,
                output_size,
                max_rotation,
                min_scale,
                max_scale,
                scale_type,
                max_translate,
                scale_aware_sigma=cfg.DATASET.SCALE_AWARE_SIGMA
            ),
            T.RandomHorizontalFlip(coco_flip_index, output_size, flip),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    return transforms
