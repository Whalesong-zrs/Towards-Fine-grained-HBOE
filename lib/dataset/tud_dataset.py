from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json
import cv2
import pickle as pkl
import torch
import random
import numpy as np

import torch.utils.data as data

import sys
sys.path.append(".")

from utils.transforms import get_affine_transform
from utils.transforms import laplacian

logger = logging.getLogger(__name__)

def load_pkl(pkl_path):
    with open(pkl_path + '.pkl', 'rb') as f:
        return pkl.load(f)

class Tud_Dataset(data.Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        # print(self.image_width)
        # print(self.image_height)nag
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.root = root
        self.is_train = is_train

        if is_train:
            # path:any_root/tud_dataset/tud_label_pkl/train_tud.pkl
            train_pkl_path = os.path.join(self.root, 'tud_dataset/tud_label_pkl/', 'train_tud_new')
            dataset_1 = load_pkl(train_pkl_path)
            val_pkl_path = os.path.join(self.root, 'tud_dataset/tud_label_pkl/', 'val_tud_new')
            dataset_2 = load_pkl(val_pkl_path)

            # render_pkl_path = os.path.join(self.root, 'render_dataset/person_label_pkl/', 'train')
            # dataset_3 = load_pkl(render_pkl_path)

            # self.tud_dataset = dataset_1 + dataset_2 + dataset_3

            self.tud_dataset = dataset_1 + dataset_2
        else:
            test_pkl_path =  os.path.join(self.root, 'tud_dataset/tud_label_pkl/', 'test_tud_new')
            dataset_3 = load_pkl(test_pkl_path)
            self.tud_dataset = dataset_3

        self.flip = cfg.DATASET.FLIP
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.hoe_sigma = cfg.DATASET.HOE_SIGMA
        self.transform = transform
    def _box2cs(self, box):
        x1, y1, x2, y2 = box[:]
        center = np.zeros((2), dtype=np.float32)
        center[0] = (x1 + x2) /2
        center[1] = (y1 + y2) /2
        w = x2 - x1
        h = y2 - y1
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25
        return center, scale
    
    def load_obj(self, index):
        # img_path:train/index.jpg
        img_path = self.tud_dataset[index]['filename']
        bbox = self.tud_dataset[index]['bboxes'][0]
        degree = self.tud_dataset[index]['degrees'][0]
        # degree = degree - 90
        # if degree < 0:
        #     degree += 360
        c,s = self._box2cs(bbox)

        return img_path, c, s, degree

    def __len__(self):
        return len(self.tud_dataset)

    def __getitem__(self, index):
        img_path, c, s, degree = self.load_obj(index)
        image_path = os.path.join(self.root, img_path)

        
        img = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            logger.error('=> fail to read {}'.format(img_path))
            raise ValueError('Fail to read {}'.format(img_path))
        
        # Flip left and right
        if self.is_train:
            sf = self.scale_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            if self.flip and random.random() <= 0.5:
                img = img[:, ::-1, :]
                c[0] = img.shape[1] - c[0] - 1

                degree = 360 - degree
        
        value_degree = degree
        degree = round(degree / 5.0)
        if degree == 72:
            degree = 0
        
        degree_heatmap = laplacian(degree, 72)
        trans = get_affine_transform(c, s, 0, self.image_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        if self.transform:
            input = self.transform(input)
        input = input.float()

        meta = {
            'image_path': img_path,
            'center': c,
            'scale': s,
            'val_dgree': value_degree,
        }
        return input,0, 0, degree_heatmap, meta
    
# for debug

if __name__ == '__main__':
    import sys
    sys.path.append(".")
    import argparse
    from config import cfg
    from config import update_config
    import torchvision.transforms as transforms
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cfg = '../experiments/lrle-3.yaml'
    args.opts, args.modelDir, args.logDir, args.dataDir = "", "", "", ""
    update_config(cfg, args)

    normalize = transforms.Normalize(
        mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]
    )
    train_dataset = Render_Person_Dataset(
        cfg, 'E:/', True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        batch_size=1,
        shuffle=False,
        # num_workers=cfg.WORKERS,
        num_workers=0,
        pin_memory=cfg.PIN_MEMORY
    )
    count = 0
    for i, b in enumerate(train_loader):
        if count == 0:
            # print(i, b)
            print(b)

            # print(b[-2].shape)
            # print(torch.max(b[0]))
            break