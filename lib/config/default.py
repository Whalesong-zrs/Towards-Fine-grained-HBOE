from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256

_C.MODEL.NAME = 'OEFormer'
_C.MODEL.oeformer = CN()
_C.MODEL.oeformer.in_channels = 3
_C.MODEL.oeformer.norm_cfg = CN()
_C.MODEL.oeformer.norm_cfg.type = 'BN2d'
_C.MODEL.oeformer.norm_cfg.requires_grad = True
_C.MODEL.oeformer.extra = CN(new_allowed=True)
_C.MODEL.oeformer.extra.drop_path_rate = 0.1
_C.MODEL.oeformer.extra.with_rpe = True

_C.MODEL.oeformer.extra.stage1 = CN(new_allowed=True)
_C.MODEL.oeformer.extra.stage1.num_modules = 1
_C.MODEL.oeformer.extra.stage1.num_branches = 1
_C.MODEL.oeformer.extra.stage1.block = 'BOTTLENECK'
_C.MODEL.oeformer.extra.stage1.num_blocks = [2]
_C.MODEL.oeformer.extra.stage1.num_channels = [64]
_C.MODEL.oeformer.extra.stage1.num_heads = [2]
_C.MODEL.oeformer.extra.stage1.num_mlp_ratios = [4]

_C.MODEL.oeformer.extra.stage2 = CN(new_allowed=True)
_C.MODEL.oeformer.extra.stage2.num_modules = 4
_C.MODEL.oeformer.extra.stage2.num_branches = 3
_C.MODEL.oeformer.extra.stage2.block = 'OEFORMERBLOCK'
_C.MODEL.oeformer.extra.stage2.num_blocks = (2, 2, 2)
_C.MODEL.oeformer.extra.stage2.num_channels = (32, 64, 128)
_C.MODEL.oeformer.extra.stage2.num_heads = [1, 2, 4]
_C.MODEL.oeformer.extra.stage2.mlp_ratios = [4, 4, 4]
_C.MODEL.oeformer.extra.stage2.window_sizes = [7, 7, 7]

_C.MODEL.oeformer.extra.stage3 = CN(new_allowed=True)
_C.MODEL.oeformer.extra.stage3.num_modules = 4
_C.MODEL.oeformer.extra.stage3.num_branches = 3
_C.MODEL.oeformer.extra.stage3.block = 'OEFORMERBLOCK'
_C.MODEL.oeformer.extra.stage3.num_blocks = (2, 2, 2)
_C.MODEL.oeformer.extra.stage3.num_channels = (32, 64, 128)
_C.MODEL.oeformer.extra.stage3.num_heads = [1, 2, 4]
_C.MODEL.oeformer.extra.stage3.mlp_ratios = [4, 4, 4]
_C.MODEL.oeformer.extra.stage3.window_sizes = [7, 7, 7]

_C.MODEL.oeformer.extra.stage4 = CN(new_allowed=True)
_C.MODEL.oeformer.extra.stage4.num_modules = 2
_C.MODEL.oeformer.extra.stage4.num_branches = 4
_C.MODEL.oeformer.extra.stage4.block = 'OEFORMERBLOCK'
_C.MODEL.oeformer.extra.stage4.num_blocks = (2, 2, 2, 2)
_C.MODEL.oeformer.extra.stage4.num_channels = (32, 64, 128, 256)
_C.MODEL.oeformer.extra.stage4.num_heads = [1, 2, 4, 8]
_C.MODEL.oeformer.extra.stage4.mlp_ratios = [4, 4, 4, 4]
_C.MODEL.oeformer.extra.stage4.window_sizes = [7, 7, 7, 7]
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.NUM_JOINTS=17
_C.LOSS = CN()
_C.LOSS.USE_ONLY_HOE = True

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.TRAIN_ROOT = ''
_C.DATASET.TRAIN_ROOT_2 = ''
_C.DATASET.VAL_ROOT = ''
_C.DATASET.VAL_ROOT_2 = ''
_C.DATASET.DATASET = 'mpii'
_C.DATASET.DATASET_2 = 'mpii'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.WINDOW_SIZE = 9
_C.DATASET.USE_RENDER = True
_C.DATASET.MIX_RATIO = 1.0

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.COLOR_RGB = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.USE_GT_BBOX = False
_C.TEST.MODEL_FILE = ''





def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

