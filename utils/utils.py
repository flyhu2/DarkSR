import sys
import torch
import numpy
import math
import rawpy

from skimage.metrics import structural_similarity
from skimage.color import rgb2ycbcr
from importlib import import_module
from config.config import args
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
from copy import deepcopy
import torch.nn as nn

import pandas as pd
import os

def creat_info_csv():
    pd.DataFrame(columns=['Contents','Value']).to_csv(os.path.join('./experiments', args.s_experiment_name,'dynamic_info.csv'), index=False)
    save_list = ['Cur_Epoc','Cur_Iter','Cur_Metr','Bes_Metr','Epo Elap','All Elap','Los TrLd']
    for i in range(len(save_list)):
        save_resolution_csv = pd.DataFrame([save_list[i]])#将列表转换为二维形式，再转化为DataFrame
        save_resolution_csv.to_csv(os.path.join('./experiments', args.s_experiment_name,'dynamic_info.csv'),mode='a',header=False,index=False)#mode指定为追加，不覆盖之前的内容
    return

def add_to_csv(index,data):

    df = pd.read_csv(os.path.join('./experiments', args.s_experiment_name,'dynamic_info.csv'))
    df = df.astype(object)  # 将DataFrame的数据类型转换为object
    df.iat[index,1] = str(data)
    df.to_csv(os.path.join('./experiments', args.s_experiment_name,'dynamic_info.csv'), index=False)
    return

def creat_metr_csv():
    save_csv =  pd.DataFrame(columns=['File No'.ljust(10),'CPSNR'.ljust(10),'R-PSNR'.ljust(10),'G-PSNR'.ljust(10),'B-PSNR'.ljust(10),'SSIM'.ljust(10),'delta E'.ljust(10),'LIPIPS'.ljust(10)])
    save_csv.to_csv(os.path.join('./experiments', args.s_experiment_name,'save_metr.csv'),index=False)
    return 

def save_to_csv(save_list):

    save_csv = pd.DataFrame([save_list])#将列表转换为二维形式，再转化为DataFrame
    save_csv.to_csv(os.path.join('./experiments', args.s_experiment_name,'save_metr.csv'),mode='a',header=False,index=False)#mode指定为追加，不覆盖之前的内容

def matrix_multiplier(matrix, img):# apply matrix img_out = matrix × img_in
        import numpy as np
        shape = img.shape
        h = shape[0]
        w = shape[1]
        out = np.zeros([h, w, 3])
        out[:, :, 0] =  matrix[0, 0] * img[:, :, 0] + \
                        matrix[0, 1] * img[:, :, 1] + \
                        matrix[0, 2] * img[:, :, 2]

        out[:, :, 1] =  matrix[1, 0] * img[:, :, 0] + \
                        matrix[1, 1] * img[:, :, 1] + \
                        matrix[1, 2] * img[:, :, 2]

        out[:, :, 2] =  matrix[2, 0] * img[:, :, 0] + \
                        matrix[2, 1] * img[:, :, 1] + \
                        matrix[2, 2] * img[:, :, 2]
        return out/np.ptp(out)

def crop_dual_test(train_img, isp_img):
    device = torch.device('cpu' if args.b_cpu else 'cuda')
    train_data = []
    isp_data = []
    scale = args.scale
    crop_size = args.n_crop_size
    test_step = crop_size // 2
    h = train_img.shape[2] // 2 * 2
    w = train_img.shape[3] // 2 * 2
    isp_img = isp_img[:, :, :h, :w]
    train_img = train_img[:, :, :h, :w]
    mask = torch.zeros(1, 3, scale * h, scale * w, device=device, dtype=torch.float)
    for i in range(0, h - crop_size + test_step, test_step):
        for j in range(0, w - crop_size + test_step, test_step):
            i = min(h - crop_size, i)
            j = min(w - crop_size, j)
            ie = i + crop_size
            je = j + crop_size
            mask[:, :, scale * i:scale * ie, scale * j:scale * je] += 1
            t_data = train_img[:, :, i:ie, j:je]
            i_data = isp_img[:, :, i:ie, j:je]
            train_data.append(t_data)
            isp_data.append(i_data)
    return train_data, isp_data, h, w, mask

def crop_single_test(train_img):
    device = torch.device('cpu' if args.b_cpu else 'cuda')
    train_data = []
    scale = args.scale
    crop_size = args.n_crop_size
    test_step = crop_size // 2
    h = train_img.shape[2] // 2 * 2
    w = train_img.shape[3] // 2 * 2
    train_img = train_img[:, :, :h, :w]
    mask = torch.zeros(1, 3, scale * h, scale * w, device=device, dtype=torch.float)
    for i in range(0, h - crop_size + test_step, test_step):
        for j in range(0, w - crop_size + test_step, test_step):
            i = min(h - crop_size, i)
            j = min(w - crop_size, j)
            ie = i + crop_size
            je = j + crop_size
            mask[:, :, scale * i:scale * ie, scale * j:scale * je] += 1
            t_data = train_img[:, :, i:ie, j:je]
            train_data.append(t_data)
    return train_data, h, w, mask

def cat_test(crop_img_list, h, w, mask):
    device = torch.device('cpu' if args.b_cpu else 'cuda')
    scale = args.scale
    crop_size = args.n_crop_size
    test_step = crop_size // 2
    res = torch.zeros([1, 3, scale * h, scale * w], device=device, dtype=torch.float)
    index = 0
    for i in range(0, scale * (h - crop_size + test_step), scale * test_step):
        for j in range(0, scale * (w - crop_size + test_step), scale * test_step):
            i = min(scale * (h - crop_size), i)
            j = min(scale * (w - crop_size), j)
            ie = i + scale * crop_size
            je = j + scale * crop_size
            res[:, :, i:ie, j:je] += crop_img_list[index][:, :, :ie - i, :je - j]
            index += 1
    res = res / mask
    return res

# RGGB 4通道HR转为3通道HR，即G=(G1+G2)/2
def RGGB2RGB(img):
    rggb = numpy.zeros([img.shape[0], img.shape[1], 3], dtype=float)
    rggb[:, :, 0] = img[:, :, 0]
    rggb[:, :, 1] = (img[:, :, 1] + img[:, :, 2])/2
    rggb[:, :, 2] = img[:, :, 3]
    return rggb

def psnr(input, target, rgb_range):
    r_input, g_input, b_input = input.split(1, 1)
    if target.shape[1] == 3:
        r_target, g_target, b_target = target.split(1, 1)
    if target.shape[1] == 4:
        r_target, g_target1, g_target2, b_target = target.split(1, 1)
        g_target = (g_target1 + g_target2)/2

    mse_r = (r_input - r_target).pow(2).mean()
    mse_g = (g_input - g_target).pow(2).mean()
    mse_b = (b_input - b_target).pow(2).mean()

    cpsnr = 10 * (rgb_range * rgb_range / ((mse_r + mse_g + mse_b) / 3)).log10()

    psnr = torch.tensor([[10 * (rgb_range * rgb_range / mse_r).log10(),
                         10 * (rgb_range * rgb_range / mse_g).log10(),
                         10 * (rgb_range * rgb_range / mse_b).log10(),
                         cpsnr]]).float()
    return psnr

def deltaE(out_data, target):
    import cv2
    import numpy as np
    out = out_data[0, :].permute(1, 2, 0).cpu().numpy()
    target = target[0, :].permute(1, 2, 0).cpu().numpy()

    lab1 = cv2.cvtColor((out*255).astype(np.uint8), cv2.COLOR_RGB2Lab)
    lab2 = cv2.cvtColor((target*255).astype(np.uint8), cv2.COLOR_RGB2Lab)
    # Calculate the difference in Lab color space
    delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))
    delta_E = np.mean(delta_e)
    delta_E = torch.tensor([[delta_E]]).float()

    return delta_E

def ssim(input, target, rgb_range):
    y1 = rgb2ycbcr(input)[:, :, 0]
    if target.shape[2] == 3:
        y2 = rgb2ycbcr(target)[:, :, 0]
    if target.shape[2] == 4:

        y2 = rgb2ycbcr(RGGB2RGB(target))[:, :, 0]

    c_s = structural_similarity(y1, y2, data_range=rgb_range, sigma=1.5, gaussian_weights=True,
                                use_sample_covariance=False)

    return torch.tensor([[c_s]]).float()

def psnr1(input, target, rgb_range):
    r_input, g_input, b_input = input.split(1, 1)
    if target.shape[1] == 3:
        r_target, g_target, b_target = target.split(1, 1)
    if target.shape[1] == 4:
        r_target, g_target1, g_target2, b_target = target.split(1, 1)
        g_target = (g_target1 + g_target2)/2

    mse_r = (r_input - r_target).pow(2).mean()
    mse_g = (g_input - g_target).pow(2).mean()
    mse_b = (b_input - b_target).pow(2).mean()
    r_psnr = 10 * (rgb_range * rgb_range / mse_r).log10()
    g_psnr = 10 * (rgb_range * rgb_range / mse_g).log10()
    b_psnr = 10 * (rgb_range * rgb_range / mse_b).log10()
    # mse_c = ((r_input - r_target + g_input - g_target + b_input - b_target)/3).pow(2).mean()
    # cpsnr = 10 * (rgb_range * rgb_range / mse_c).log10()
    # cpsnr = 10 * (rgb_range * rgb_range / ((mse_r + mse_g + mse_b) / 3)).log10()

    psnr = torch.tensor([[r_psnr, g_psnr, b_psnr,
                          (r_psnr+g_psnr+b_psnr)/3,]]).float()

    return psnr


def ssim1(input, target, rgb_range):
    c_s1 = structural_similarity(input[:, :, 0], target[:, :, 0], data_range=rgb_range, sigma=1.5, gaussian_weights=True,
                                use_sample_covariance=False)
    c_s2 = structural_similarity(input[:, :, 1], target[:, :, 1], data_range=rgb_range, sigma=1.5, gaussian_weights=True,
                                use_sample_covariance=False)
    c_s3 = structural_similarity(input[:, :, 2], target[:, :, 2], data_range=rgb_range, sigma=1.5, gaussian_weights=True,
                                use_sample_covariance=False)

    return torch.tensor([[(c_s1+c_s2+c_s3)/3]]).float()


def ssim2(image1, image2, K, window_size, L):
    _, channel1, _, _ = image1.size()
    _, channel2, _, _ = image2.size()
    channel = min(channel1, channel2)

    # gaussian window generation
    sigma = 1.5  # default
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    # define constants
    # * L = 255 for constants doesn't produce meaningful results; thus L = 1
    # C1 = (K[0]*L)**2;
    # C2 = (K[1]*L)**2;
    C1 = K[0] ** 2
    C2 = K[1] ** 2

    mu1 = F.conv2d(image1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(image2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image1 * image1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(image2 * image2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(image1 * image2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def calc_para(net):
    num_params = 0
    f_params = 0
    m_params = 0
    l_params = 0
    stage = 1
    total_str = 'The number of parameters for each sub-block:\n'

    for param in net.parameters():
        num_params += param.numel()

# 计算网络各部分参数量
    for body in net.named_children():
        res_params = 0
        res_str = []
        for param in body[1].parameters():
            res_params += param.numel()
        res_str = '[{:s}] parameters: {}\n'.format(body[0], res_params)
        total_str = total_str + res_str
        if stage == 1:
            f_params = f_params + res_params
            # if body[0] == 'base_detail':
            #     stage = 2
        elif stage == 2:
            m_params = m_params + res_params
            # if body[0] == 'conv2d':
            #     stage = 3
        elif stage == 3:
            l_params = l_params + res_params
        if 'anchor' in body[0]:     stage += 1

    total_str = total_str + '[total] parameters: {}\n\n'.format(num_params) + \
                '[first_net]\tparameters: {:.3f} M\n'.format(f_params/1e6) + \
                '[middle_net]parameters: {:.3f} M\n'.format(m_params/1e6) + \
                '[last_net]\tparameters: {:.3f} M\n'.format(l_params/1e6) + \
                '[total_net]\tparameters: {:.3f} M\n'.format(num_params/1e6) + \
                '**'
    return total_str


def import_fun(fun_dir, module):
    fun = module.split('.')
    m = import_module(fun_dir + '.' + fun[0])
    return getattr(m, fun[1])


def catch_exception(exception):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print('{}: {}.'.format(exc_type, exception), exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno)


def alloc_gpumem():
    # noinspection PyBroadException
    ram = []
    try:
        device = torch.device('cpu' if args.b_cpu else 'cuda')
        while True:
            ram.append(torch.randn((int(0.5 * 1024), 1024, 32, 32), device=device, requires_grad=False))
    except Exception as e:
        pass


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

