#
# check_point.py
#
# This file is part of LIDN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version

import datetime
import os
import matplotlib.pyplot as plt
import torch

from utils import utils
from config.config import args

plt.switch_backend('agg')


class CheckPoint(object):
    def __init__(self, filepath, experiment_name):
        self.loss = torch.Tensor().cpu()
        self.result = torch.Tensor().cpu()
        self.experiment_name = experiment_name
        self.start_epoch = 0
        self.color = ('red', 'green', 'blue', 'black')
        self.psnr_label = ('R_PSNR', 'G_PSNR', 'B_PSNR', 'CPSNR')
        self.label = 'result on {}'.format(self.experiment_name)

        self.filepath = os.path.join(filepath, self.experiment_name)
        os.makedirs(self.filepath, exist_ok=True)
        os.makedirs(os.path.join(self.filepath, 'model'), exist_ok=True)
        os.makedirs(os.path.join(self.filepath, 'result'), exist_ok=True)

    # def save(self, model, loss, loss_value, optimizer, scheduler, epoch, is_best=False, result=None):
    def save(self, model, loss, loss_value, optimizer, epoch, is_best=False, result=None): # lr_cos没有scheduler
        state = {'model': model.state_dict(),
                 'loss': loss.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 # 'scheduler': scheduler.state_dict(), # lr_cos没有scheduler
                 'epoch': epoch}

        torch.save(state, os.path.join(self.filepath, 'model', 'model_latest.pth'))
        if is_best:
            torch.save(state, os.path.join(self.filepath, 'model', 'model_best.pth'))
        if args.b_save_all_models:
            torch.save(state, os.path.join(self.filepath, 'model', 'model{}.pth'.format(epoch)))

        self.loss = torch.cat((self.loss, loss_value.unsqueeze(0).cpu()))
        axis = torch.linspace(epoch - self.loss.shape[0] + 1, epoch, self.loss.shape[0])

        # noinspection PyBroadException
        try:
            fig, axes = plt.subplots(dpi=600)

            for i, l in enumerate(loss.get_loss()):
                axes.plot(axis.numpy(), self.loss[:, i].numpy(), label=l.get('type'),
                          c=self.color[i % len(self.color)], linewidth=0.5)
            axes.set_title(self.label)
            axes.set_xlabel('Epochs')
            axes.set_ylabel('Loss')
            axes.legend()
            axes.grid(True, linestyle=':', linewidth=0.5)

            fig.savefig(os.path.join(self.filepath, 'losses_{}.png'.format(self.experiment_name)))
            plt.cla()
            plt.close()
        except Exception as e:
            utils.catch_exception(e)

        if result is not None and len(result) != 0:
            if self.start_epoch == 0:
                self.start_epoch = epoch

            self.result = torch.cat((self.result, result))
            axis = torch.linspace(self.start_epoch, epoch, self.result.shape[0])

            # noinspection PyBroadException
            try:
                fig, axes = plt.subplots(dpi=600)

                for i, (c, l) in enumerate(zip(self.color, self.psnr_label)):
                    axes.plot(axis.numpy(), self.result[:, i].numpy(), label=l, c=c, linewidth=0.5)
                axes.set_title(self.label)
                axes.set_xlabel('Epochs')
                axes.set_ylabel('PSNR')
                axes.legend()
                axes.grid(True, linestyle=':', linewidth=0.5)
                fig.savefig(os.path.join(self.filepath, 'results_{}.png'.format(self.experiment_name)))
                plt.cla()
                plt.close()
            except Exception as e:
                utils.catch_exception(e)

    def load(self, to_device, is_best=False):
        model_name = 'model_best.pth'
        return torch.load(os.path.join(self.filepath, 'model', model_name), map_location=to_device)

    # def load_restoration(self, to_device, is_best=False):
    #     model_name = 'model_best.pth' if is_best else 'model_latest.pth'
    #     return torch.load(os.path.join(self.filepath, 'restoration_model', model_name), map_location=to_device)

    def save_config(self, args, model):
        with open(os.path.join(self.filepath, 'config.txt'), 'w') as f:
            f.write(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '\n\n')
            print(model, file=f)
            f.write('\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))

    def load_pre_model(self, to_device):
        # model_name = 'model_best.pth' if is_best else 'model_latest.pth'
        model_name = args.pretrain_dm_models
        return torch.load(os.path.join('pretrained_models', model_name, 'model', 'model_best.pth'), map_location=to_device)
