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
import random
sys.path.append(".")

from utils.transforms import get_affine_transform
from utils.transforms import laplacian
logger = logging.getLogger(__name__)

def load_pkl(pkl_path):
    with open(pkl_path + '.pkl', 'rb') as f:
        return pkl.load(f)

class Mebow_Dataset(data.Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.root = root
        self.is_train = is_train

        if is_train:
            train_pkl_path = os.path.join(self.root, 'mebow_dataset/mebow_label_pkl/', 'train')
            dataset = load_pkl(train_pkl_path)

            train_pkl_path2 = os.path.join(self.root, 'render_dataset/person_label_pkl/', 'train')
            dataset_2 = load_pkl(train_pkl_path2)

            if not cfg.DATASET.USE_RENDER:
                self.mebow_dataset = dataset
            elif cfg.DATASET.USE_RENDER:
                mix_ratio = cfg.DATASET.MIX_RATIO
                len_render = int(mix_ratio*len(dataset_2))
                self.mebow_dataset = dataset + dataset_2[:len_render]
            
            random.shuffle(self.mebow_dataset)
            
        else:
            val_pkl_path = os.path.join(self.root, 'mebow_dataset/mebow_label_pkl/', 'val')
            dataset = load_pkl(val_pkl_path)
            self.mebow_dataset = dataset
        self.flip = cfg.DATASET.FLIP 
        self.scale_factor = cfg.DATASET.SCALE_FACTOR 
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.window_size = cfg.DATASET.WINDOW_SIZE
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
        # img_path:mebow_dataset/mebow_image/train_2017/' + image_id + '.jpg
        img_path = self.mebow_dataset[index]['filename']
        bbox = self.mebow_dataset[index]['bboxes'][0]
        degree = self.mebow_dataset[index]['degrees'][0]
        c,s = self._box2cs(bbox)
        return img_path, c, s, degree

    def __len__(self):
        return len(self.mebow_dataset)

    def __getitem__(self, index):
        img_path, c, s, degree = self.load_obj(index)
        img_path = os.path.join(self.root, img_path)
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
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
        degree = round(degree/5.0)
        if degree == 72:
            degree = 0
        

        degree_heatmap = laplacian(degree, 72, self.window_size)
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
        return input, 0, 0, degree_heatmap, meta
    
