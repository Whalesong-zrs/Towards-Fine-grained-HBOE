
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
from torch import distributed as dist
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

from utils.utils import get_optimizer
from utils.utils import save_checkpoint
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
    parser.add_argument('--local_rank', default=0, type=int) 
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    seed_everything(12770)
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")  
    torch.cuda.set_device(args.local_rank)
    
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

   
    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    ).cuda(args.local_rank)

    # if args.local_rank == 0:
    logger, final_output_dir, tb_log_dir = create_logger(
    cfg, args.cfg, 'train')
    writer_dict = {
    'writer': SummaryWriter(logdir=tb_log_dir),
    'train_global_steps': 0,
    'valid_global_steps': 0,
    }
    

    pre_model = cfg.MODEL.PRETRAINED
    
    model.init_weights(pre_model)
    model = model.cuda(args.local_rank)

    if args.local_rank == 0:
        
        print('world_size', torch.distributed.get_world_size()) 
        print('=> creating {}'.format(cfg.OUTPUT_DIR))
        print('=> creating {}'.format(final_output_dir))
        print('=> creating {}'.format(tb_log_dir))
        logger.info(pprint.pformat(args))
        logger.info(cfg)
        
        logger.info(get_model_summary(model, dump_input))

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, 
                                                     device_ids=[args.local_rank], 
                                                     output_device=args.local_rank, 
                                                     find_unused_parameters=True, 
                                                     broadcast_buffers=False)
   
    
    

    criterions = {}
    criterions['hoe_loss'] = torch.nn.SmoothL1Loss(beta=0.2).cuda()
   
    
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.TRAIN_ROOT, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
   

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.VAL_ROOT, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    # former
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )


    best_perf = 200.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    checkpoint_file ='checkpoint.pth'

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
     
        train_sampler.set_epoch(epoch)

        train(cfg, train_loader, train_dataset, model, criterions, optimizer, epoch,
            final_output_dir, tb_log_dir, writer_dict, args.local_rank)
        
        if args.local_rank == 0:
            perf_indicator = validate(
                cfg, valid_loader, valid_dataset, model, criterions,
                final_output_dir, tb_log_dir, writer_dict
            )
        
        
            if perf_indicator <= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            lr_scheduler.step()
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            logger.info('best_model{}'.format(best_perf))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': best_perf,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

        dist.barrier()

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )

    if args.local_rank == 0:
        logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
        torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
