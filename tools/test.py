
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config

from core.function import train
from core.function import validate

from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.utils import seed_everything

import dataset
from model_arch.oeformer import OEFormer
from model_arch.hrnet_with_head import PoseHighResolutionNet

def parse_args():
    parser = argparse.ArgumentParser(description='Towards fine-grained HBOE')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    seed_everything(12770)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.MODEL.NAME == 'oeformer':
        oeformer_cfg = cfg.MODEL.oeformer
        model = OEFormer(
            in_channels=oeformer_cfg['in_channels'],
            norm_cfg=oeformer_cfg['norm_cfg'],
            extra=oeformer_cfg['extra'],
        )
    elif cfg.MODEL.NAME == 'hrnet_head':
        model = PoseHighResolutionNet(cfg, is_train=False)
    else:
        raise ValueError("No Match Model Name. cfg.MODEL.NAME must in ['oeformer', 'hrnet_head']")

    this_dir = os.path.dirname(__file__)

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    dump_input = dump_input.cuda()
    model = model.cuda()
    logger.info(get_model_summary(model, dump_input))
    model.init_weights(cfg.TEST.MODEL_FILE)
    

    criterions = {}
    
    criterions['hoe_loss'] = torch.nn.SmoothL1Loss(beta=0.2).cuda()
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.VAL_ROOT, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=8,
    )

    validate(
        cfg, valid_loader, valid_dataset, model, criterions,
        final_output_dir, tb_log_dir, save_pickle=False, draw_pic=False
    )

if __name__ == '__main__':
    main()
