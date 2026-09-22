# -*- coding: utf-8 -*-
from os import path as osp
import cv2
import numpy as np
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.data.bsrgan_degradation import degradation_bsrgan
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class BSRGANDataset(data.Dataset):
    """Dataset used for BSRGAN model:
    Designing a Practical Degradation Model for Deep Blind Image Super-Resolution (ICCV 2021).

    It reads GT images and dynamically synthesizes practical LQ images on the fly.

    Args:
        opt (dict): Config for train datasets.
    """

    def __init__(self, opt):
        super(BSRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.scale = opt.get('scale', 4)
        self.gt_size = opt.get('gt_size', 48)
        self.lq_patchsize = max(8, self.gt_size // self.scale)

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.gt_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)  # float32 [0, 1], BGR

        # Random horizontal flip / rotation
        img_gt = augment(img_gt, self.opt.get('use_hflip', False), self.opt.get('use_rot', False))

        # Convert to RGB for BSRGAN degradation pipeline
        img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        # Apply BSRGAN degradation to synthesize LQ and crop
        img_lq_rgb, img_gt_crop_rgb = degradation_bsrgan(img_gt_rgb, sf=self.scale, lq_patchsize=self.lq_patchsize)

        # Convert back to BGR
        img_lq = cv2.cvtColor(img_lq_rgb, cv2.COLOR_RGB2BGR)
        img_gt_crop = cv2.cvtColor(img_gt_crop_rgb, cv2.COLOR_RGB2BGR)

        # numpy to tensor: (H, W, C) -> (C, H, W)
        img_gt_tensor, img_lq_tensor = img2tensor([img_gt_crop, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq_tensor,
            'gt': img_gt_tensor,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
