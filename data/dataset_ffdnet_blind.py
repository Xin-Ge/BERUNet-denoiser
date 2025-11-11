import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import torch.nn.functional as F
from math import ceil

class DatasetFFDNet_Blind(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H/M for denosing on AWGN with a range of sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., FFDNet, H = f(L, sigma), sigma is noise level
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetFFDNet_Blind, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma      = opt['sigma'] if opt['sigma'] else [0, 75]
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else 25
        self.image_divisor = opt['image_divisor'] if opt['image_divisor'] else 1
        self.image_extension = opt['image_extension'] if opt['image_extension'] else 0
        self.dataset_enlarge_ratio = opt['dataset_enlarge_ratio'] if opt['dataset_enlarge_ratio'] else 1
        self.isclip = opt['isclip'] if opt['isclip'] else False
        self.isquantization = opt['isquantization'] if opt['isquantization'] else False
        
        # -------------------------------------
        # get the path of H, return None if input is None
        # -------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        print('Preload {} training images into memory, dataset enlarge ratio is {}:'.format(len(self.paths_H),self.dataset_enlarge_ratio))
        self.images_H = []
        for idx, H_path in enumerate(self.paths_H):
            if idx%100 == 0:
                print('Read image: {}/{}'.format(idx+1,len(self.paths_H)))
            img_H = util.imread_uint(H_path, self.n_channels)
            self.images_H.append(img_H)

        self.paths_H  = self.paths_H * self.dataset_enlarge_ratio
        self.images_H = self.images_H * self.dataset_enlarge_ratio

    def __getitem__(self, index):
        # -------------------------------------
        # get H image
        # -------------------------------------
        H_path = self.paths_H[index]
        # img_H = util.imread_uint(H_path, self.n_channels)
        img_H  = self.images_H[index]

        L_path = H_path

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

            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            # ---------------------------------
            # get noise level
            # ---------------------------------
            # noise_level = torch.FloatTensor([np.random.randint(self.sigma_min, self.sigma_max)])/255.0
            noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)])/255.0

            # ---------------------------------
            # add noise
            # ---------------------------------
            noise = torch.randn(img_L.size()).mul_(noise_level).float()
            img_L.add_(noise)

            if self.isclip:
                img_L = img_L.clamp(0, 1)

            if self.isquantization:
                img_L = util.tensor2uint(img_L)
                img_L = util.uint2tensor3(img_L)

        else:
            """
            # --------------------------------
            # get L/H/sigma image pairs
            # --------------------------------
            """
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)
            noise_level = torch.FloatTensor([self.sigma_test/255.0])

            if self.isclip:
                img_L = img_L.clip(0, 1)

            if self.isquantization:
                img_L = util.single2uint(img_L)
                img_L = util.uint2single(img_L)
            # ---------------------------------
            # L/H image pairs
            # ---------------------------------
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        noise_level = noise_level.unsqueeze(1).unsqueeze(1)

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
