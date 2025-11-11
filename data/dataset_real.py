import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import os


class DatasetRealSIDD(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H/M for denosing on AWGN with a range of sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., FFDNet, H = f(L, sigma), sigma is noise level
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetRealSIDD, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if opt['H_size'] else 64
        self.dataset_enlarge_ratio = opt['dataset_enlarge_ratio'] if opt['dataset_enlarge_ratio'] else 1
        self.gt_dir    = 'groundtruth'
        self.noisy_dir = 'noisy'

        # -------------------------------------
        # get the path of H, return None if input is None
        # -------------------------------------
        self.paths_H  = util.get_image_paths(os.path.join(opt['dataroot_H'], self.gt_dir))
        print('Preload {} training images into memory:'.format(len(self.paths_H)))
        self.images_H = []
        self.paths_L  = []
        self.images_L = []
        
        for idx, _ in enumerate(self.paths_H):
            if idx%10 == 0:
                print('Read image: {}/{}'.format(idx+1,len(self.paths_H)))

            file_name = os.path.basename(self.paths_H[idx])  # 获取文件名
            new_file_name = file_name.replace("GT", "NOISY")
            new_file_path = os.path.join(opt['dataroot_H'], self.noisy_dir, new_file_name)
            self.paths_L.append(new_file_path)
            
            img_H = util.imread_uint(self.paths_H[idx], self.n_channels)
            self.images_H.append(img_H)
            
            img_L = util.imread_uint(self.paths_L[idx], self.n_channels)
            self.images_L.append(img_L)
                        
        self.paths_L  = self.paths_L  * self.dataset_enlarge_ratio
        self.images_L = self.images_L * self.dataset_enlarge_ratio
        self.paths_H  = self.paths_H  * self.dataset_enlarge_ratio
        self.images_H = self.images_H * self.dataset_enlarge_ratio

    def __getitem__(self, index):
        # -------------------------------------
        # get H image
        # -------------------------------------
        H_path, L_path = self.paths_H[index] , self.paths_L[index]
        img_H , img_L  = self.images_H[index], self.images_L[index]

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H/M patch pairs
            # --------------------------------
            """
            H, W = img_H.shape[:2]

            # ---------------------------------
            # randomly crop the patch
            # ---------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            mode = random.randint(0, 7)
            patch_H, patch_L = util.augment_img(patch_H, mode=mode), util.augment_img(patch_L, mode=mode)

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_H, img_L = util.uint2tensor3(patch_H), util.uint2tensor3(patch_L)

        else:
            """
            # --------------------------------
            # get L/H/sigma image pairs
            # --------------------------------
            """

            # ---------------------------------
            # L/H image pairs
            # ---------------------------------
            img_H, img_L = util.uint2tensor3(img_H), util.uint2tensor3(img_L)


        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
