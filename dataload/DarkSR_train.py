import os
import glob
import torch
import numpy as np
from dataload import DarkSR_base
import random

class DarkSR_train(DarkSR_base.DarkSR_base):
    def __init__(self, args, name='TRAINING', b_train=True, b_benchmark=False):
        self.ext = ('npy',)
        self.dir_root = os.path.join(args.dir_dataset, name)
        self.dir_label = os.path.join(self.dir_root, 'GT_ISP')  # XsRGB HR
        if args.input_type == 'dual':
            self.dir_data = os.path.join(self.dir_root, 'LR_RAW')  # Xraw LR, DNG format floder is: LR_RAW
            self.dir_isp = os.path.join(self.dir_root, 'LR_ISP')  # Xref LR
        elif args.input_type == 'raw':
            self.dir_data = os.path.join(self.dir_root, 'LR_RAW')
        elif args.input_type == 'srgb':
            self.dir_data = os.path.join(self.dir_root, 'LR_SRGB')
        self.data_range = args.train_range  # train/test data range
        super(DarkSR_train, self).__init__(args, name=name, b_train=b_train, b_benchmark=b_benchmark)
    
    def pack3(self,img):
        # pack RGGB Bayer RAW to 3 channels [RGB]
        C,H,W = img.shape
        raw = np.zeros([3,H,W],dtype = np.float32)
        raw[0, 0::2, 0::2] = img[:,0::2, 0::2]      #R
        raw[1, 0::2, 1::2] = img[:,0::2, 1::2]     #G1
        raw[1, 1::2, 0::2] = img[:,1::2, 0::2]     #G2
        raw[2, 1::2, 1::2] = img[:,1::2, 1::2]    #B
        # (1,H,W) -> (3,H,W)
        return raw

    def __getitem__(self, index):

        if self.args.input_type == 'dual':
            data = np.expand_dims(np.load(self.name_data[index]),axis=0)/65535
            isp = np.load(self.name_isp[index])/255
        elif self.args.input_type == 'raw':
            data = np.expand_dims(np.load(self.name_data[index]),axis=0)/65535
            data = self.pack3(data)
        elif self.args.input_type == 'srgb':
            isp = np.load(self.name_isp[index])/255

        label = np.load(self.name_label[index])/255

        if self.b_train:
            if self.args.input_type == 'dual':
                _, iw, ih = data.shape
                ix = random.randrange(0, iw - (self.args.n_patch_size) + 1, 2)    # 随机偶数
                iy = random.randrange(0, ih - (self.args.n_patch_size) + 1, 2)
                data = data[:,ix:ix + self.args.n_patch_size, iy:iy + self.args.n_patch_size]
                isp = isp[ix:ix + self.args.n_patch_size, iy:iy + self.args.n_patch_size, :]
                label = label[ix*2:(ix + self.args.n_patch_size)*2, iy*2:(iy + self.args.n_patch_size)*2, :]

                im_data = np.ascontiguousarray(data) #(1,H,W)
                im_isp = np.ascontiguousarray(isp.transpose((2, 0, 1))) #(3,H,W)
                im_label = np.ascontiguousarray(label.transpose((2, 0, 1))) #(3,2H,2W)
                return torch.from_numpy(im_data).float(), torch.from_numpy(im_isp).float(), torch.from_numpy(im_label).float() # 
            elif self.args.input_type == 'raw':
                _, iw, ih = data.shape
                ix = random.randrange(0, iw - (self.args.n_patch_size) + 1, 2)
                iy = random.randrange(0, ih - (self.args.n_patch_size) + 1, 2)
                data = data[:,ix:ix + self.args.n_patch_size, iy:iy + self.args.n_patch_size]
                label = label[ix*2:(ix + self.args.n_patch_size)*2, iy*2:(iy + self.args.n_patch_size)*2, :]
                im_data = np.ascontiguousarray(data) #(1,H,W)
                im_label = np.ascontiguousarray(label.transpose((2, 0, 1)))
                return torch.from_numpy(im_data).float(), torch.from_numpy(im_label).float()
            elif self.args.input_type == 'srgb':
                iw, ih = isp.shape[:2]
                ix = random.randrange(0, iw - (self.args.n_patch_size) + 1, 2)
                iy = random.randrange(0, ih - (self.args.n_patch_size) + 1, 2)
                isp = isp[ix:ix + self.args.n_patch_size, iy:iy + self.args.n_patch_size, :]
                label = label[ix*2:(ix + self.args.n_patch_size)*2, iy*2:(iy + self.args.n_patch_size)*2, :]
                im_isp = np.ascontiguousarray(isp.transpose((2, 0, 1)))
                im_label = np.ascontiguousarray(label.transpose((2, 0, 1)))
                return torch.from_numpy(im_isp).float(), torch.from_numpy(im_label).float()
        

