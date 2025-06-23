#
# common.py
#
# This file is part of LIDN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version

import os
import torch
import numpy as np
import cv2
import time
from config.config import args
from utils import log, timer, utils
import math
from utils.utils import save_to_csv

epochs = 0
iteration = 0

def adjust_learning_rate(optimizer, current_iter, lr, max_iteration, switch_iteration):
    # """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr_max = lr
    lr_min = 1e-6 # 1e-6
    if current_iter < switch_iteration:
        lr = lr_min + lr_max * current_iter / switch_iteration
    else:
        lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * (current_iter - switch_iteration) / (max_iteration - switch_iteration))) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_epoch():
    global epochs
    return epochs
def dual_train(model, data_loader, criterion, optimizer, model_ema):
    global epochs, iteration
    epochs = epochs + 1

    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    device = torch.device('cpu' if args.b_cpu else 'cuda')  # ('cpu' if args.b_cpu else 'cuda:{}'.format(args.n_index_gpu))

    # # loss.step()
    model.train()

    timer_train = timer.Timer()
    timer_load_data = timer.Timer(True)
    display_losses = torch.zeros(len(criterion.get_loss()), device=device, requires_grad=False)
    load_data_elapsed_ticks = 0
    timer_train_elapsed_ticks = 0
    max_iteration = len(data_loader.dataset) // args.n_batch_size * args.n_epochs  # 1
    switch_iteration = len(data_loader.dataset) // args.n_batch_size * args.lrdecay_how_epos

    # trained_num = 0
    for iteration_batch, (data, isp, sRGB) in enumerate(data_loader):
        # trained_num = trained_num + data_loader.batch_sampler.batch_size
        data, isp, sRGB = data.to(device, non_blocking=True),\
                          isp.to(device, non_blocking=True),\
                          sRGB.to(device, non_blocking=True)

        timer_load_data.stop()
        load_data_elapsed_ticks += timer_load_data.elapsed_ticks()

        adjust_learning_rate(optimizer, iteration, args.lr, max_iteration, switch_iteration)  # 2

        utils.add_to_csv(1,'[{}/{}] [Lr: {:.4e}] [Time: ({})]'.format(
                            iteration + 1,
                            max_iteration,
                            optimizer.param_groups[0]['lr'],  # scheduler.get_last_lr()[0],
                            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                        ))

        timer_train.restart()

        optimizer.zero_grad()
        model_out = model(data, isp)  # model_out可能是个列表，例如可视化特征，以及多个监督信息
        epoch_loss = criterion(model_out, sRGB) # 计算loss
        epoch_loss.backward()
        if args.gclip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.gclip)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.gclip)

        optimizer.step()
        if args.use_ema:
            model_ema.update(model)  # 更新模型参数

        timer_train.stop()
        timer_train_elapsed_ticks += timer_train.elapsed_ticks()

        for i, l in enumerate(criterion.get_loss()):
            display_losses[i] += l.get('value')
            # if math.isnan(l.get('value')):
            #     print(data_loader.dataset.name_isp[iteration_batch])
            # if math.isinf(l.get('value')):
            #     print(data_loader.dataset.name_isp[iteration_batch])
            # if l.get('value') > 0.2:
            #     print(data_loader.dataset.name_isp[iteration_batch])
            #     print(l.get('value'))

        if (iteration_batch + 1) % args.n_batches_per_print == 0 or (iteration_batch + 1) * len(data) == len(data_loader.dataset):
            display_loss = ''
            for i, l in enumerate(criterion.get_loss()):
                display_loss = display_loss + '[{}: {:.4f}]'.format(
                    l.get('type'), display_losses[i] / (iteration_batch + 1))

            utils.add_to_csv(6,'[Iter: {}/{}\t][Loss: {}\t] [Tr/Ld: {:.1f}s/{:.1f}s]'.format(
                                (iteration_batch + 1) * len(data),
                                len(data_loader.dataset),
                                display_loss,
                                timer_train_elapsed_ticks,
                                load_data_elapsed_ticks
                            ))
                            
            load_data_elapsed_ticks = 0
            timer_train_elapsed_ticks = 0

        iteration += 1  # 3
        timer_load_data.restart()
    return display_losses / (len(data_loader.dataset) / args.n_batch_size)
def dual_test(model, data_loader):
    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    device = torch.device('cpu' if args.b_cpu else 'cuda')

    model.eval()
    timer_test = timer.Timer()
    timer_test_elapsed_ticks = 0

    with torch.no_grad():
        im_psnr = torch.Tensor().to(device)
        im_ssim = torch.Tensor().to(device)
        im_psnr1 = torch.Tensor().to(device)
        im_ssim1 = torch.Tensor().to(device)
        im_delta_e = torch.Tensor().to(device)
        im_delta_e1 = torch.Tensor().to(device)
        timer_test_elapsed_ticks = 0

        import shutil  
        shutil.rmtree(os.path.join('./experiments', args.s_experiment_name, 'result'))  
        os.mkdir(os.path.join('./experiments', args.s_experiment_name, 'result')) 
        
        timer_test = timer.Timer()
        for d_index, d in enumerate(data_loader):
            t_psnr = torch.Tensor().to(device)
            t_ssim = torch.Tensor().to(device)
            t_psnr1 = torch.Tensor().to(device)
            t_ssim1 = torch.Tensor().to(device)
            t_delta_e = torch.Tensor().to(device)
            t_delta_e1 = torch.Tensor().to(device)
            for batch_index, (data, isp, sRGB) in enumerate(d):
                #传数据到GPU
                data, isp, sRGB = data.to(device, non_blocking=True), \
                                isp.to(device, non_blocking=True), \
                                sRGB.to(device, non_blocking=True)
                try:
                    target = sRGB
                    
                    crop_output = []
                    #对测试图片裁剪，raw和isp图像对
                    crop_data, crop_isp, h, w, mask = utils.crop_dual_test(data, isp)
                    timer_test.restart()
                    #对batch每一对计算
                    for i in range(len(crop_data)):
                        crop_model_out = model(crop_data[i], crop_isp[i])
                        #模型输出有多个时，选择第一个
                        crop_model_rgb = crop_model_out if len(crop_model_out) == 1 else crop_model_out[0]  
                        #数据限制范围0~1
                        crop_model_rgb = crop_model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                        #一个batch存为一个列表
                        crop_output.append(crop_model_rgb)
                    timer_test.restart()
                    #model_out = model(data, isp)
                    model_out = utils.cat_test(crop_output, h, w, mask)
                    timer_test.stop()

                    timer_test_elapsed_ticks += timer_test.elapsed_ticks()
                    model_rgb = model_out if len(model_out) == 1 else model_out[0]#模型输出有多个时，选择第一个
                    model_rgb = model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                    # out_lr = lr.mul(1.0).clamp(0, args.n_rgb_range)

                    all_psnr = utils.psnr(model_rgb, target, args.n_rgb_range).to(device)
                    im_psnr = torch.cat((im_psnr, all_psnr))
                    t_psnr = torch.cat((t_psnr, all_psnr))

                    all_psnr1 = utils.psnr1(model_rgb, target, args.n_rgb_range).to(device)
                    im_psnr1 = torch.cat((im_psnr1, all_psnr1))
                    t_psnr1 = torch.cat((t_psnr1, all_psnr1))

                    # all_ssim = ssim.ssim(model_rgb.to(torch.uint8), target.to(torch.uint8), args.n_rgb_range).to(device)
                    # im_ssim = torch.cat((im_ssim, all_ssim.unsqueeze(0)))
                    # t_ssim = torch.cat((t_ssim, all_ssim.unsqueeze(0)))

                    out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()
                    out_label = target[0, :].permute(1, 2, 0).cpu().numpy()
                    # out_lr = out_lr[0, :].permute(1, 2, 0).cpu().numpy()

                    if args.n_rgb_range == 255:
                        out_data = np.uint8(out_data)
                        out_label = np.uint8(out_label)
                        # out_lr = np.uint8(out_lr)
                    elif args.n_rgb_range == 65535:
                        out_data = np.uint16(out_data)
                        out_label = np.uint16(out_label)
                        # out_lr = np.uint8(out_lr)

                    # all_ssim1 = utils.ssim(out_data, out_label, 1.0).to(device)
                    # all_ssim2 = utils.ssim(out_data * 255.0, out_label * 255.0, 255.0).to(device)
                    all_ssim = utils.ssim(np.uint8(out_data * 255), np.uint8(out_label * 255), 255).to(device) # 为了测试SSIM，改动的地方！！！
                    im_ssim = torch.cat((im_ssim, all_ssim))
                    t_ssim = torch.cat((t_ssim, all_ssim))

                    all_ssim1 = utils.ssim1(out_data, out_label, args.n_rgb_range).to(device)
                    im_ssim1 = torch.cat((im_ssim1, all_ssim1))
                    t_ssim1 = torch.cat((t_ssim1, all_ssim1))

                    all_delta_e = utils.deltaE(model_rgb, target).to(device)
                    im_delta_e = torch.cat((im_delta_e, all_delta_e))
                    t_delta_e = torch.cat((t_delta_e, all_delta_e))

                    all_delta_e1 = utils.deltaE(model_rgb, target).to(device)
                    im_delta_e1 = torch.cat((im_delta_e1, all_delta_e1))
                    t_delta_e1 = torch.cat((t_delta_e1, all_delta_e1))
                    # 添加输出每张图片的PSNR和SSIM
                    filename, _ = os.path.splitext(os.path.basename(d.dataset.name_data[batch_index]))
                    info_log.write('{}_{}_{}_{}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}, {:.5f}'.format(
                            d.dataset.name,
                            args.s_model.split('.')[0],
                            batch_index,
                            filename,
                            all_psnr[:, -1].item(),
                            all_psnr[:, 0].item(),
                            all_psnr[:, 1].item(),
                            all_psnr[:, 2].item(),
                            all_ssim.item(),
                            all_delta_e.item(),
                    ))

                    if args.b_save_results:
                        filename, _ = os.path.splitext(os.path.basename(d.dataset.name_data[batch_index]))
                        path = os.path.join('./experiments', args.s_experiment_name, 'result',
                                            filename + '_' + args.s_model.split('.')[0] + '_' +
                                            '_PSNR_'+'{:.3f}'.format(all_psnr[:, -1].item())+
                                            '_SSIM_'+'{:.5f}'.format(all_ssim.item())+
                                            '_deltaE_'+'{:.5f}'.format(all_delta_e.item())+ '.bmp')
                        cv2.imwrite(path, cv2.cvtColor(np.uint8(out_data * 255), cv2.COLOR_RGB2BGR))
                        save_list = [filename.ljust(10),
                                     '{:.3f}'.format(all_psnr[:, -1].item()).ljust(10),
                                     '{:.3f}'.format(all_psnr[:, 0].item()).ljust(10),
                                     '{:.3f}'.format(all_psnr[:, 1].item()).ljust(10),
                                     '{:.3f}'.format(all_psnr[:, 2].item()).ljust(10),
                                     '{:.5f}'.format(all_ssim.item()).ljust(10),
                                     '{:.5f}'.format(all_delta_e.item()).ljust(10)]
                        save_to_csv(save_list)
                        # np.save(path, out_data)
                except Exception as e:
                    utils.catch_exception(e)

            t_psnr = t_psnr.mean(dim=0, keepdim=True)
            t_ssim = t_ssim.mean(dim=0, keepdim=True)
            t_psnr1 = t_psnr1.mean(dim=0, keepdim=True)
            t_ssim1 = t_ssim1.mean(dim=0, keepdim=True)
            t_delta_e = t_delta_e.mean(dim=0, keepdim=True)
            t_delta_e1 = t_delta_e1.mean(dim=0, keepdim=True)
            
            info_log.write('my   {}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f},{:.5f}'.format(
                d.dataset.name,
                t_psnr[:, -1].item(),
                t_psnr[:, 0].item(),
                t_psnr[:, 1].item(),
                t_psnr[:, 2].item(),
                t_ssim.item(),
                t_delta_e.item(),
            ))

            save_list1 = ['MY'.ljust(10),
                        '{:.3f}'.format(t_psnr[:, -1].item()).ljust(10),
                        '{:.3f}'.format(t_psnr[:, 0].item()).ljust(10),
                        '{:.3f}'.format(t_psnr[:, 1].item()).ljust(10),
                        '{:.3f}'.format(t_psnr[:, 2].item()).ljust(10),
                        '{:.5f}'.format(t_ssim.item()).ljust(10),
                        '{:.5f}'.format(t_delta_e.item()).ljust(10)]
            save_to_csv(save_list1)

            info_log.write('mean {}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f},{:.5f}'.format(
                d.dataset.name,
                t_psnr1[:, -1].item(),
                t_psnr1[:, 0].item(),
                t_psnr1[:, 1].item(),
                t_psnr1[:, 2].item(),
                t_ssim1.item(),
                t_delta_e1.item(),  
            ))
            save_list2 = ['MEAN'.ljust(10),
                        '{:.3f}'.format(t_psnr1[:, -1].item()).ljust(10),
                        '{:.3f}'.format(t_psnr1[:, 0].item()).ljust(10),
                        '{:.3f}'.format(t_psnr1[:, 1].item()).ljust(10),
                        '{:.3f}'.format(t_psnr1[:, 2].item()).ljust(10),
                        '{:.5f}'.format(t_ssim1.item()).ljust(10),
                        '{:.5f}'.format(t_delta_e1.item()).ljust(10)]
            save_to_csv(save_list2)

        im_psnr = im_psnr.mean(dim=0, keepdim=True)
        im_ssim1 = im_ssim1.mean(dim=0, keepdim=True)
        im_delta_e1 = im_delta_e1.mean(dim=0, keepdim=True)

    return im_psnr, im_ssim1,im_delta_e1, timer_test_elapsed_ticks
def raw_train(model, data_loader, criterion, optimizer, model_ema):
    global epochs, iteration
    epochs = epochs + 1

    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    device = torch.device('cpu' if args.b_cpu else 'cuda')  # ('cpu' if args.b_cpu else 'cuda:{}'.format(args.n_index_gpu))

    # # loss.step()
    model.train()

    timer_train = timer.Timer()
    timer_load_data = timer.Timer(True)
    display_losses = torch.zeros(len(criterion.get_loss()), device=device, requires_grad=False)
    load_data_elapsed_ticks = 0
    timer_train_elapsed_ticks = 0
    max_iteration = len(data_loader.dataset) // args.n_batch_size * args.n_epochs  # 1
    switch_iteration = len(data_loader.dataset) // args.n_batch_size * 5

    # trained_num = 0
    for iteration_batch, (data, sRGB) in enumerate(data_loader):
        # trained_num = trained_num + data_loader.batch_sampler.batch_size
        data, sRGB = data.to(device, non_blocking=True),sRGB.to(device, non_blocking=True)

        timer_load_data.stop()
        load_data_elapsed_ticks += timer_load_data.elapsed_ticks()

        adjust_learning_rate(optimizer, iteration, args.lr, max_iteration, switch_iteration)  # 2
        utils.add_to_csv(1,'[{}/{}] [Lr: {:.4e}] [Time: ({})]'.format(
            iteration + 1,
            max_iteration,
            optimizer.param_groups[0]['lr'],  # scheduler.get_last_lr()[0],
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        ))

        timer_train.restart()

        optimizer.zero_grad()
        model_out = model(data)  # model_out是个列表
        epoch_loss = criterion(model_out, sRGB) # 原
        epoch_loss.backward()
        if args.gclip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.gclip)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.gclip)

        optimizer.step()
        if args.use_ema:
            model_ema.update(model)  # 更新模型参数

        timer_train.stop()
        timer_train_elapsed_ticks += timer_train.elapsed_ticks()

        for i, l in enumerate(criterion.get_loss()):
            display_losses[i] += l.get('value')
            # if math.isnan(l.get('value')):
            #     print(data_loader.dataset.name_isp[iteration_batch])
            # if math.isinf(l.get('value')):
            #     print(data_loader.dataset.name_isp[iteration_batch])
            # if l.get('value') > 0.2:
            #     print(data_loader.dataset.name_isp[iteration_batch])
            #     print(l.get('value'))

        if (iteration_batch + 1) % args.n_batches_per_print == 0 or (iteration_batch + 1) * len(data) == len(data_loader.dataset):
            display_loss = ''
            for i, l in enumerate(criterion.get_loss()):
                display_loss = display_loss + '[{}: {:.4f}]'.format(
                    l.get('type'), display_losses[i] / (iteration_batch + 1))

            utils.add_to_csv(6,'[Iter: {}/{}\t][Loss: {}\t] [Tr/Ld: {:.1f}s/{:.1f}s]'.format(
                (iteration_batch + 1) * len(data),
                len(data_loader.dataset),
                display_loss,
                timer_train_elapsed_ticks,
                load_data_elapsed_ticks
            ))
            load_data_elapsed_ticks = 0
            timer_train_elapsed_ticks = 0

        iteration += 1  # 3
        timer_load_data.restart()
    return display_losses / (len(data_loader.dataset) / args.n_batch_size)
def raw_test(model, data_loader):
    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    device = torch.device('cpu' if args.b_cpu else 'cuda')

    model.eval()
    timer_test = timer.Timer()
    timer_test_elapsed_ticks = 0

    with torch.no_grad():
        im_psnr = torch.Tensor().to(device)
        im_ssim = torch.Tensor().to(device)
        im_psnr1 = torch.Tensor().to(device)
        im_ssim1 = torch.Tensor().to(device)
        im_delta_e = torch.Tensor().to(device)
        im_delta_e1 = torch.Tensor().to(device)
        timer_test_elapsed_ticks = 0

        import shutil  
        shutil.rmtree(os.path.join('./experiments', args.s_experiment_name, 'result'))  
        os.mkdir(os.path.join('./experiments', args.s_experiment_name, 'result')) 
        
        timer_test = timer.Timer()
        for d_index, d in enumerate(data_loader):
            t_psnr = torch.Tensor().to(device)
            t_ssim = torch.Tensor().to(device)
            t_psnr1 = torch.Tensor().to(device)
            t_ssim1 = torch.Tensor().to(device)
            t_delta_e = torch.Tensor().to(device)
            t_delta_e1 = torch.Tensor().to(device)
            for batch_index, (data, sRGB) in enumerate(d):
                #传数据到GPU
                data, sRGB = data.to(device, non_blocking=True), sRGB.to(device, non_blocking=True)
                try:
                    target = sRGB
                    
                    crop_output = []
                    #对测试图片裁剪，raw和isp图像对
                    crop_data,  h, w, mask = utils.crop_single_test(data)
                    timer_test.restart()
                    #对batch每一对计算
                    for i in range(len(crop_data)):
                        crop_model_out = model(crop_data[i])
                        #模型输出有多个时，选择第一个
                        crop_model_rgb = crop_model_out if len(crop_model_out) == 1 else crop_model_out[0]  
                        #数据限制范围0~1
                        crop_model_rgb = crop_model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                        #一个batch存为一个列表
                        crop_output.append(crop_model_rgb)
                    timer_test.restart()
                    #model_out = model(data)
                    model_out = utils.cat_test(crop_output, h, w, mask)
                    timer_test.stop()

                    timer_test_elapsed_ticks += timer_test.elapsed_ticks()
                    model_rgb = model_out if len(model_out) == 1 else model_out[0]#模型输出有多个时，选择第一个
                    model_rgb = model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                    # out_lr = lr.mul(1.0).clamp(0, args.n_rgb_range)

                    all_psnr = utils.psnr(model_rgb, target, args.n_rgb_range).to(device)
                    im_psnr = torch.cat((im_psnr, all_psnr))
                    t_psnr = torch.cat((t_psnr, all_psnr))

                    all_psnr1 = utils.psnr1(model_rgb, target, args.n_rgb_range).to(device)
                    im_psnr1 = torch.cat((im_psnr1, all_psnr1))
                    t_psnr1 = torch.cat((t_psnr1, all_psnr1))

                    # all_ssim = ssim.ssim(model_rgb.to(torch.uint8), target.to(torch.uint8), args.n_rgb_range).to(device)
                    # im_ssim = torch.cat((im_ssim, all_ssim.unsqueeze(0)))
                    # t_ssim = torch.cat((t_ssim, all_ssim.unsqueeze(0)))

                    out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()
                    out_label = target[0, :].permute(1, 2, 0).cpu().numpy()
                    # out_lr = out_lr[0, :].permute(1, 2, 0).cpu().numpy()

                    if args.n_rgb_range == 255:
                        out_data = np.uint8(out_data)
                        out_label = np.uint8(out_label)
                        # out_lr = np.uint8(out_lr)
                    elif args.n_rgb_range == 65535:
                        out_data = np.uint16(out_data)
                        out_label = np.uint16(out_label)
                        # out_lr = np.uint8(out_lr)

                    # all_ssim1 = utils.ssim(out_data, out_label, 1.0).to(device)
                    # all_ssim2 = utils.ssim(out_data * 255.0, out_label * 255.0, 255.0).to(device)
                    all_ssim = utils.ssim(np.uint8(out_data * 255), np.uint8(out_label * 255), 255).to(device) # 为了测试SSIM，改动的地方！！！
                    im_ssim = torch.cat((im_ssim, all_ssim))
                    t_ssim = torch.cat((t_ssim, all_ssim))

                    all_ssim1 = utils.ssim1(out_data, out_label, args.n_rgb_range).to(device)
                    im_ssim1 = torch.cat((im_ssim1, all_ssim1))
                    t_ssim1 = torch.cat((t_ssim1, all_ssim1))

                    all_delta_e = utils.deltaE(model_rgb, target).to(device)
                    im_delta_e = torch.cat((im_delta_e, all_delta_e))
                    t_delta_e = torch.cat((t_delta_e, all_delta_e))

                    all_delta_e1 = utils.deltaE(model_rgb, target).to(device)
                    im_delta_e1 = torch.cat((im_delta_e1, all_delta_e1))
                    t_delta_e1 = torch.cat((t_delta_e1, all_delta_e1))
                    # 添加输出每张图片的PSNR和SSIM
                    filename, _ = os.path.splitext(os.path.basename(d.dataset.name_data[batch_index]))
                    info_log.write('{}_{}_{}_{}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}, {:.5f}'.format(
                            d.dataset.name,
                            args.s_model.split('.')[0],
                            batch_index,
                            filename,
                            all_psnr[:, -1].item(),
                            all_psnr[:, 0].item(),
                            all_psnr[:, 1].item(),
                            all_psnr[:, 2].item(),
                            all_ssim.item(),
                            all_delta_e.item(),
                    ))

                    if args.b_save_results:
                        filename, _ = os.path.splitext(os.path.basename(d.dataset.name_data[batch_index]))
                        path = os.path.join('./experiments', args.s_experiment_name, 'result',
                                            filename + '_' + args.s_model.split('.')[0] + '_' +
                                            '_PSNR_'+'{:.3f}'.format(all_psnr[:, -1].item())+
                                            '_SSIM_'+'{:.5f}'.format(all_ssim.item())+
                                            '_deltaE_'+'{:.5f}'.format(all_delta_e.item())+ '.bmp')
                        cv2.imwrite(path, cv2.cvtColor(np.uint8(out_data * 255), cv2.COLOR_RGB2BGR))
                    
                    save_list = [filename.ljust(10),
                                '{:.3f}'.format(all_psnr[:, -1].item()).ljust(10),
                                '{:.3f}'.format(all_psnr[:, 0].item()).ljust(10),
                                '{:.3f}'.format(all_psnr[:, 1].item()).ljust(10),
                                '{:.3f}'.format(all_psnr[:, 2].item()).ljust(10),
                                '{:.5f}'.format(all_ssim.item()).ljust(10),
                                '{:.5f}'.format(all_delta_e.item()).ljust(10)]
                    save_to_csv(save_list)

                except Exception as e:
                    utils.catch_exception(e)

            t_psnr = t_psnr.mean(dim=0, keepdim=True)
            t_ssim = t_ssim.mean(dim=0, keepdim=True)
            t_psnr1 = t_psnr1.mean(dim=0, keepdim=True)
            t_ssim1 = t_ssim1.mean(dim=0, keepdim=True)
            t_delta_e = t_delta_e.mean(dim=0, keepdim=True)
            t_delta_e1 = t_delta_e1.mean(dim=0, keepdim=True)
            
            info_log.write('my   {}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f},{:.5f}'.format(
                d.dataset.name,
                t_psnr[:, -1].item(),
                t_psnr[:, 0].item(),
                t_psnr[:, 1].item(),
                t_psnr[:, 2].item(),
                t_ssim.item(),
                t_delta_e.item(),
            ))

            save_list1 = ['MY'.ljust(10),
                        '{:.3f}'.format(t_psnr[:, -1].item()).ljust(10),
                        '{:.3f}'.format(t_psnr[:, 0].item()).ljust(10),
                        '{:.3f}'.format(t_psnr[:, 1].item()).ljust(10),
                        '{:.3f}'.format(t_psnr[:, 2].item()).ljust(10),
                        '{:.5f}'.format(t_ssim.item()).ljust(10),
                        '{:.5f}'.format(t_delta_e.item()).ljust(10)]
            save_to_csv(save_list1)

            info_log.write('mean {}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f},{:.5f}'.format(
                d.dataset.name,
                t_psnr1[:, -1].item(),
                t_psnr1[:, 0].item(),
                t_psnr1[:, 1].item(),
                t_psnr1[:, 2].item(),
                t_ssim1.item(),
                t_delta_e1.item(),  
            ))
            save_list2 = ['MEAN'.ljust(10),
                        '{:.3f}'.format(t_psnr1[:, -1].item()).ljust(10),
                        '{:.3f}'.format(t_psnr1[:, 0].item()).ljust(10),
                        '{:.3f}'.format(t_psnr1[:, 1].item()).ljust(10),
                        '{:.3f}'.format(t_psnr1[:, 2].item()).ljust(10),
                        '{:.5f}'.format(t_ssim1.item()).ljust(10),
                        '{:.5f}'.format(t_delta_e1.item()).ljust(10)]
            save_to_csv(save_list2)

        im_psnr = im_psnr.mean(dim=0, keepdim=True)
        im_ssim1 = im_ssim1.mean(dim=0, keepdim=True)
        im_delta_e1 = im_delta_e1.mean(dim=0, keepdim=True)

    return im_psnr, im_ssim1,im_delta_e1, timer_test_elapsed_ticks
def srgb_train(model, data_loader, criterion, optimizer, model_ema):
    global epochs, iteration
    epochs = epochs + 1

    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    device = torch.device('cpu' if args.b_cpu else 'cuda')  # ('cpu' if args.b_cpu else 'cuda:{}'.format(args.n_index_gpu))

    # # loss.step()
    model.train()

    timer_train = timer.Timer()
    timer_load_data = timer.Timer(True)
    display_losses = torch.zeros(len(criterion.get_loss()), device=device, requires_grad=False)
    load_data_elapsed_ticks = 0
    timer_train_elapsed_ticks = 0
    max_iteration = len(data_loader.dataset) // args.n_batch_size * args.n_epochs  # 1
    switch_iteration = len(data_loader.dataset) // args.n_batch_size * 5

    # trained_num = 0
    for iteration_batch, (isp, sRGB) in enumerate(data_loader):
        # trained_num = trained_num + data_loader.batch_sampler.batch_size
        isp, sRGB = isp.to(device, non_blocking=True),sRGB.to(device, non_blocking=True)

        timer_load_data.stop()
        load_data_elapsed_ticks += timer_load_data.elapsed_ticks()

        adjust_learning_rate(optimizer, iteration, args.lr, max_iteration, switch_iteration)  # 2
        utils.add_to_csv(1,'[{}/{}] [Lr: {:.4e}] [Time: ({})]'.format(
            iteration + 1,
            max_iteration,
            optimizer.param_groups[0]['lr'],  # scheduler.get_last_lr()[0],
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        ))

        timer_train.restart()

        optimizer.zero_grad()
        model_out = model(isp)  # model_out可能是个列表
        epoch_loss = criterion(model_out, sRGB) # 原
        epoch_loss.backward()
        if args.gclip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.gclip)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.gclip)

        optimizer.step()
        if args.use_ema:
            model_ema.update(model)  # 更新模型参数

        timer_train.stop()
        timer_train_elapsed_ticks += timer_train.elapsed_ticks()

        for i, l in enumerate(criterion.get_loss()):
            display_losses[i] += l.get('value')
            # if math.isnan(l.get('value')):
            #     print(data_loader.dataset.name_isp[iteration_batch])
            # if math.isinf(l.get('value')):
            #     print(data_loader.dataset.name_isp[iteration_batch])
            # if l.get('value') > 0.2:
            #     print(data_loader.dataset.name_isp[iteration_batch])
            #     print(l.get('value'))

        if (iteration_batch + 1) % args.n_batches_per_print == 0 or (iteration_batch + 1) * len(isp) == len(data_loader.dataset):
            display_loss = ''
            for i, l in enumerate(criterion.get_loss()):
                display_loss = display_loss + '[{}: {:.4f}]'.format(
                    l.get('type'), display_losses[i] / (iteration_batch + 1))

            utils.add_to_csv(6,'[Iter: {}/{}\t][Loss: {}\t] [Tr/Ld: {:.1f}s/{:.1f}s]'.format(
                (iteration_batch + 1) * len(isp),
                len(data_loader.dataset),
                display_loss,
                timer_train_elapsed_ticks,
                load_data_elapsed_ticks
            ))
            load_data_elapsed_ticks = 0
            timer_train_elapsed_ticks = 0

        iteration += 1  # 3
        timer_load_data.restart()
    return display_losses / (len(data_loader.dataset) / args.n_batch_size)
def srgb_test(model, data_loader):
    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    device = torch.device('cpu' if args.b_cpu else 'cuda')

    model.eval()
    timer_test = timer.Timer()
    timer_test_elapsed_ticks = 0

    with torch.no_grad():
        im_psnr = torch.Tensor().to(device)
        im_ssim = torch.Tensor().to(device)
        im_psnr1 = torch.Tensor().to(device)
        im_ssim1 = torch.Tensor().to(device)
        im_delta_e = torch.Tensor().to(device)
        im_delta_e1 = torch.Tensor().to(device)
        timer_test_elapsed_ticks = 0

        import shutil  
        shutil.rmtree(os.path.join('./experiments', args.s_experiment_name, 'result'))  
        os.mkdir(os.path.join('./experiments', args.s_experiment_name, 'result')) 
        
        timer_test = timer.Timer()
        for d_index, d in enumerate(data_loader):
            t_psnr = torch.Tensor().to(device)
            t_ssim = torch.Tensor().to(device)
            t_psnr1 = torch.Tensor().to(device)
            t_ssim1 = torch.Tensor().to(device)
            t_delta_e = torch.Tensor().to(device)
            t_delta_e1 = torch.Tensor().to(device)
            for batch_index, (isp, sRGB) in enumerate(d):
                #传数据到GPU
                isp, sRGB = isp.to(device, non_blocking=True), sRGB.to(device, non_blocking=True)
                try:
                    target = sRGB
                    
                    crop_output = []
                    #对测试图片裁剪，raw和isp图像对
                    crop_isp, h, w, mask = utils.crop_single_test(isp)
                    timer_test.restart()
                    #对batch每一对计算
                    for i in range(len(crop_isp)):
                        crop_model_out = model(crop_isp[i])
                        #模型输出有多个时，选择第一个
                        crop_model_rgb = crop_model_out if len(crop_model_out) == 1 else crop_model_out[0]  
                        #数据限制范围0~1
                        crop_model_rgb = crop_model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                        #一个batch存为一个列表
                        crop_output.append(crop_model_rgb)
                    timer_test.restart()
                    #model_out = model(isp)
                    model_out = utils.cat_test(crop_output, h, w, mask)
                    timer_test.stop()

                    timer_test_elapsed_ticks += timer_test.elapsed_ticks()
                    model_rgb = model_out if len(model_out) == 1 else model_out[0]#模型输出有多个时，选择第一个
                    model_rgb = model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                    # out_lr = lr.mul(1.0).clamp(0, args.n_rgb_range)

                    all_psnr = utils.psnr(model_rgb, target, args.n_rgb_range).to(device)
                    im_psnr = torch.cat((im_psnr, all_psnr))
                    t_psnr = torch.cat((t_psnr, all_psnr))

                    all_psnr1 = utils.psnr1(model_rgb, target, args.n_rgb_range).to(device)
                    im_psnr1 = torch.cat((im_psnr1, all_psnr1))
                    t_psnr1 = torch.cat((t_psnr1, all_psnr1))

                    # all_ssim = ssim.ssim(model_rgb.to(torch.uint8), target.to(torch.uint8), args.n_rgb_range).to(device)
                    # im_ssim = torch.cat((im_ssim, all_ssim.unsqueeze(0)))
                    # t_ssim = torch.cat((t_ssim, all_ssim.unsqueeze(0)))

                    out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()
                    out_label = target[0, :].permute(1, 2, 0).cpu().numpy()
                    # out_lr = out_lr[0, :].permute(1, 2, 0).cpu().numpy()

                    if args.n_rgb_range == 255:
                        out_data = np.uint8(out_data)
                        out_label = np.uint8(out_label)
                        # out_lr = np.uint8(out_lr)
                    elif args.n_rgb_range == 65535:
                        out_data = np.uint16(out_data)
                        out_label = np.uint16(out_label)
                        # out_lr = np.uint8(out_lr)

                    # all_ssim1 = utils.ssim(out_data, out_label, 1.0).to(device)
                    # all_ssim2 = utils.ssim(out_data * 255.0, out_label * 255.0, 255.0).to(device)
                    all_ssim = utils.ssim(np.uint8(out_data * 255), np.uint8(out_label * 255), 255).to(device) # 为了测试SSIM，改动的地方！！！
                    im_ssim = torch.cat((im_ssim, all_ssim))
                    t_ssim = torch.cat((t_ssim, all_ssim))

                    all_ssim1 = utils.ssim1(out_data, out_label, args.n_rgb_range).to(device)
                    im_ssim1 = torch.cat((im_ssim1, all_ssim1))
                    t_ssim1 = torch.cat((t_ssim1, all_ssim1))

                    all_delta_e = utils.deltaE(model_rgb, target).to(device)
                    im_delta_e = torch.cat((im_delta_e, all_delta_e))
                    t_delta_e = torch.cat((t_delta_e, all_delta_e))

                    all_delta_e1 = utils.deltaE(model_rgb, target).to(device)
                    im_delta_e1 = torch.cat((im_delta_e1, all_delta_e1))
                    t_delta_e1 = torch.cat((t_delta_e1, all_delta_e1))
                    # 添加输出每张图片的PSNR和SSIM
                    filename, _ = os.path.splitext(os.path.basename(d.dataset.name_isp[batch_index]))
                    info_log.write('{}_{}_{}_{}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}, {:.5f}'.format(
                            d.dataset.name,
                            args.s_model.split('.')[0],
                            batch_index,
                            filename,
                            all_psnr[:, -1].item(),
                            all_psnr[:, 0].item(),
                            all_psnr[:, 1].item(),
                            all_psnr[:, 2].item(),
                            all_ssim.item(),
                            all_delta_e.item(),
                    ))

                    if args.b_save_results:
                        filename, _ = os.path.splitext(os.path.basename(d.dataset.name_isp[batch_index]))
                        path = os.path.join('./experiments', args.s_experiment_name, 'result',
                                            filename + '_' + args.s_model.split('.')[0] + '_' +
                                            '_PSNR_'+'{:.3f}'.format(all_psnr[:, -1].item())+
                                            '_SSIM_'+'{:.5f}'.format(all_ssim.item())+
                                            '_deltaE_'+'{:.5f}'.format(all_delta_e.item())+ '.bmp')
                        cv2.imwrite(path, cv2.cvtColor(np.uint8(out_data * 255), cv2.COLOR_RGB2BGR))

                    save_list = [filename.ljust(10),
                                '{:.3f}'.format(all_psnr[:, -1].item()).ljust(10),
                                '{:.3f}'.format(all_psnr[:, 0].item()).ljust(10),
                                '{:.3f}'.format(all_psnr[:, 1].item()).ljust(10),
                                '{:.3f}'.format(all_psnr[:, 2].item()).ljust(10),
                                '{:.5f}'.format(all_ssim.item()).ljust(10),
                                '{:.5f}'.format(all_delta_e.item()).ljust(10)]
                    save_to_csv(save_list)
                except Exception as e:
                    utils.catch_exception(e)

            t_psnr = t_psnr.mean(dim=0, keepdim=True)
            t_ssim = t_ssim.mean(dim=0, keepdim=True)
            t_psnr1 = t_psnr1.mean(dim=0, keepdim=True)
            t_ssim1 = t_ssim1.mean(dim=0, keepdim=True)
            t_delta_e = t_delta_e.mean(dim=0, keepdim=True)
            t_delta_e1 = t_delta_e1.mean(dim=0, keepdim=True)
            
            info_log.write('my   {}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f},{:.5f}'.format(
                d.dataset.name,
                t_psnr[:, -1].item(),
                t_psnr[:, 0].item(),
                t_psnr[:, 1].item(),
                t_psnr[:, 2].item(),
                t_ssim.item(),
                t_delta_e.item(),
            ))

            save_list1 = ['MY'.ljust(10),
                        '{:.3f}'.format(t_psnr[:, -1].item()).ljust(10),
                        '{:.3f}'.format(t_psnr[:, 0].item()).ljust(10),
                        '{:.3f}'.format(t_psnr[:, 1].item()).ljust(10),
                        '{:.3f}'.format(t_psnr[:, 2].item()).ljust(10),
                        '{:.5f}'.format(t_ssim.item()).ljust(10),
                        '{:.5f}'.format(t_delta_e.item()).ljust(10)]
            save_to_csv(save_list1)

            info_log.write('mean {}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f},{:.5f}'.format(
                d.dataset.name,
                t_psnr1[:, -1].item(),
                t_psnr1[:, 0].item(),
                t_psnr1[:, 1].item(),
                t_psnr1[:, 2].item(),
                t_ssim1.item(),
                t_delta_e1.item(),  
            ))
            save_list2 = ['MEAN'.ljust(10),
                        '{:.3f}'.format(t_psnr1[:, -1].item()).ljust(10),
                        '{:.3f}'.format(t_psnr1[:, 0].item()).ljust(10),
                        '{:.3f}'.format(t_psnr1[:, 1].item()).ljust(10),
                        '{:.3f}'.format(t_psnr1[:, 2].item()).ljust(10),
                        '{:.5f}'.format(t_ssim1.item()).ljust(10),
                        '{:.5f}'.format(t_delta_e1.item()).ljust(10)]
            save_to_csv(save_list2)

        im_psnr = im_psnr.mean(dim=0, keepdim=True)
        im_ssim1 = im_ssim1.mean(dim=0, keepdim=True)
        im_delta_e1 = im_delta_e1.mean(dim=0, keepdim=True)

    return im_psnr, im_ssim1,im_delta_e1, timer_test_elapsed_ticks

def inference_dual_input(model):
    #使用数据集外的数据进行推理，同理对输入的RAW图像进行简单的预处理，包括黑电平校正，初始白平衡
    #然后将3通道的RAW和ISP sRGB同时送入网络进行计算，将网络输出结果直接保存即可
    #推理时，直接运行 main_inference.py
    #导入的模型和test一样在config里进行设置实验文件夹以调用对应的best pth
    #值得注意的是RAW图像和ISP的图像尺寸关系，直接使用photoshop导出的尺寸略小于raw的尺寸
    #需要使用exiftoolGUI删除默认裁剪标签，确保尺寸问题
    #相机的isp图像一般小于rawpy读取的raw图像，可能源于裁剪标签，当然也可以读取裁剪标签对raw进行裁剪而不改动isp
    #如果遇到那种删不掉裁剪标签的相机，可以先使用dng conventor转换为标准的dng格式，再使用raw2dng重新封装为dng，然后再用Photoshop打开导出为jpg
    #这样有个问题就是jpg不一定符合原始相机的风格
    device = torch.device('cpu' if args.b_cpu else 'cuda')
    base_address = './inference/input_data/'
    from glob import glob
    img_names = [os.path.basename(x) for x in sorted(glob(os.path.join(base_address,'raw', '*.*')))]
    model.eval()
    print('All files names:',img_names)
    with torch.no_grad():
        for example_num in range(len(img_names)):
            print('processing %d/%d'%(example_num+1,len(img_names)) )
            import rawpy
            print('file:%s is pre-processing...'%(img_names[example_num]))
            with rawpy.imread(os.path.join(base_address,'raw', img_names[example_num])) as raw:
                raw_data = raw.raw_image_visible
                import imageio as im
                isp = im.imread(os.path.join(base_address,'isp', img_names[example_num][:-4]+'.jpg'))
                #isp = raw.postprocess(no_auto_bright=True,user_flip=0)
                import numpy
                print('raw file shape:(%d,%d), srgb file shape:(%d,%d,%d)'%
                      (raw_data.shape[0],raw_data.shape[1],
                       isp.shape[0],isp.shape[1],isp.shape[2]))                
                black = raw.black_level_per_channel[0] 
                if raw.camera_white_level_per_channel == None:
                    white = raw.white_level
                else:
                    white = raw.camera_white_level_per_channel[0]
                wb = numpy.asarray(raw.camera_whitebalance)
                wb = wb/wb[1]
                wb = wb
                wb = numpy.array([ [wb[0],        0,            0],
                                [0,            1,            0],
                                [0,            0,        wb[2]]])
                raw_data = raw_data.astype(numpy.float32)
                lr_data = numpy.maximum(raw_data-black,0)/(white-black)
                raw_pattern = raw.raw_pattern
                new_H,new_W = raw_data.shape[0]//2*2, raw_data.shape[1]//2*2
                data = numpy.zeros((new_H,new_W,3), dtype=float)
                if raw_pattern[0,0] == 0:# RGGB
                    data[0::2, 0::2, 0] = lr_data[0:new_H:2, 0:new_W:2] #R
                    data[1::2, 0::2, 1] = lr_data[1:new_H:2, 0:new_W:2] #G
                    data[0::2, 1::2, 1] = lr_data[0:new_H:2, 1:new_W:2] #G
                    data[1::2, 1::2, 2] = lr_data[1:new_H:2, 1:new_W:2] #B
                elif raw_pattern[0,1] == 0:# GRB4new_Hnew_W
                    data[0::2, 1::2, 0] = lr_data[0:new_H:2, 1:new_W:2] #R
                    data[0::2, 0::2, 1] = lr_data[0:new_H:2, 0:new_W:2] #G
                    data[1::2, 1::2, 1] = lr_data[1:new_H:2, 1:new_W:2] #G
                    data[1::2, 0::2, 2] = lr_data[1:new_H:2, 0:new_W:2] #B            
                elif raw_pattern[1,0] == 0:# GBR4new_Hnew_W
                    data[1::2, 0::2, 0] = lr_data[1:new_H:2, 0:new_W:2] #R
                    data[0::2, 0::2, 1] = lr_data[0:new_H:2, 0:new_W:2] #G
                    data[1::2, 1::2, 1] = lr_data[1:new_H:2, 1:new_W:2] #G
                    data[0::2, 1::2, 2] = lr_data[0:new_H:2, 1:new_W:2] #B
                else:
                    data[1::2, 1::2, 0] = lr_data[1:new_H:2, 1:new_W:2] #R
                    data[1::2, 0::2, 1] = lr_data[1:new_H:2, 0:new_W:2] #G
                    data[0::2, 1::2, 1] = lr_data[0:new_H:2, 1:new_W:2] #G
                    data[0::2, 0::2, 2] = lr_data[0:new_H:2, 0:new_W:2] #B
                from utils.utils import matrix_multiplier
                data = matrix_multiplier(wb,data) 
                data = np.ascontiguousarray(data.transpose((2, 0, 1)))
                isp_data = isp[:data.shape[1]*2,:data.shape[2]*2,:]
                isp_data = np.expand_dims(isp_data,axis=0)
                data_out = np.expand_dims(data,axis=0)
                isp_data = np.ascontiguousarray(isp_data.transpose((0, 3, 1, 2)))
                data_out = torch.from_numpy(data_out).float()
                isp_data = torch.from_numpy(isp_data).float()/255

                data_out, isp_data = data_out.to(device, non_blocking=True), \
                                     isp_data.to(device, non_blocking=True), \
                                        
                try:
                    #图像分辨率太大时，对原图进行裁剪，然后对输出的结果进行拼接
                    crop_output = []
                    #对测试图片裁剪，raw和isp图像对
                    crop_data, crop_isp, h, w, mask = utils.crop_dual_test(data_out, isp_data)
                    print('file pre-process done!')
                    print('starting inference······')
                    for i in range(len(crop_data)):
                        crop_model_out = model(crop_data[i], crop_isp[i])
                        #模型输出有多个时，选择第一个
                        crop_model_rgb = crop_model_out if len(crop_model_out) == 1 else crop_model_out[0]  
                        #数据限制范围0~1
                        crop_model_rgb = crop_model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                        #一个batch存为一个列表
                        crop_output.append(crop_model_rgb)
                    #model_out = model(data_out, isp_data)
                    model_out = utils.cat_test(crop_output, h, w, mask)
                    model_rgb = model_out if len(model_out) == 1 else model_out[0]#模型输出有多个时，选择第一个
                    model_rgb = model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                    out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()

                    if args.b_save_results:
                        filename, _ = os.path.splitext(img_names[example_num])
                        path = os.path.join('./inference/result/', filename +'_'+args.s_model.split('.')[0]+ '.jpg')
                        print('saving output······')
                        cv2.imwrite(path, cv2.cvtColor(np.uint8(out_data * 255), cv2.COLOR_RGB2BGR))
                        print('--------------------------------------------------')
                except Exception as e:
                    utils.catch_exception(e)
    return print('All files done!')
def inference_raw_input(model):
    #使用数据集外的数据进行推理，同理对输入的RAW图像进行简单的预处理，包括黑电平校正，初始白平衡
    #然后将3通道的RAW和ISP sRGB同时送入网络进行计算，将网络输出结果直接保存即可
    #推理时，直接运行 main_inference.py
    #导入的模型和test一样在config里进行设置实验文件夹以调用对应的best pth
    #值得注意的是RAW图像和ISP的图像尺寸关系，直接使用photoshop导出的尺寸略小于raw的尺寸
    #需要使用exiftoolGUI删除默认裁剪标签，确保尺寸问题
    #相机的isp图像一般小于rawpy读取的raw图像，可能源于裁剪标签，当然也可以读取裁剪标签对raw进行裁剪而不改动isp
    #如果遇到那种删不掉裁剪标签的相机，可以先使用dng conventor转换为标准的dng格式，再使用raw2dng重新封装为dng，然后再用Photoshop打开导出为jpg
    #这样有个问题就是jpg不一定符合原始相机的风格
    device = torch.device('cpu' if args.b_cpu else 'cuda')
    base_address = './inference/input_data/'
    from glob import glob
    img_names = [os.path.basename(x) for x in sorted(glob(os.path.join(base_address,'raw', '*.*')))]
    model.eval()
    print('All files names:',img_names)
    with torch.no_grad():
        for example_num in range(len(img_names)):
            print('processing %d/%d'%(example_num+1,len(img_names)) )
            import rawpy
            print('file:%s is pre-processing...'%(img_names[example_num]))
            with rawpy.imread(os.path.join(base_address,'raw', img_names[example_num])) as raw:
                raw_data = raw.raw_image_visible
                print('raw file shape:(%d,%d)'%(raw_data.shape[0],raw_data.shape[1]))                 
                import numpy
                black = raw.black_level_per_channel[0] 
                if raw.camera_white_level_per_channel == None:
                    white = raw.white_level
                else:
                    white = raw.camera_white_level_per_channel[0]
                wb = numpy.asarray(raw.camera_whitebalance)
                wb = wb/wb[1]
                wb = wb
                wb = numpy.array([ [wb[0],        0,            0],
                                [0,            1,            0],
                                [0,            0,        wb[2]]])
                raw_data = raw_data.astype(numpy.float32)
                lr_data = numpy.maximum(raw_data-black,0)/(white-black)
                raw_pattern = raw.raw_pattern
                new_H,new_W = raw_data.shape[0]//2*2, raw_data.shape[1]//2*2
                data = numpy.zeros((new_H,new_W,3), dtype=float)
                if raw_pattern[0,0] == 0:# RGGB
                    data[0::2, 0::2, 0] = lr_data[0:new_H:2, 0:new_W:2] #R
                    data[1::2, 0::2, 1] = lr_data[1:new_H:2, 0:new_W:2] #G
                    data[0::2, 1::2, 1] = lr_data[0:new_H:2, 1:new_W:2] #G
                    data[1::2, 1::2, 2] = lr_data[1:new_H:2, 1:new_W:2] #B
                elif raw_pattern[0,1] == 0:# GRB4new_Hnew_W
                    data[0::2, 1::2, 0] = lr_data[0:new_H:2, 1:new_W:2] #R
                    data[0::2, 0::2, 1] = lr_data[0:new_H:2, 0:new_W:2] #G
                    data[1::2, 1::2, 1] = lr_data[1:new_H:2, 1:new_W:2] #G
                    data[1::2, 0::2, 2] = lr_data[1:new_H:2, 0:new_W:2] #B            
                elif raw_pattern[1,0] == 0:# GBR4new_Hnew_W
                    data[1::2, 0::2, 0] = lr_data[1:new_H:2, 0:new_W:2] #R
                    data[0::2, 0::2, 1] = lr_data[0:new_H:2, 0:new_W:2] #G
                    data[1::2, 1::2, 1] = lr_data[1:new_H:2, 1:new_W:2] #G
                    data[0::2, 1::2, 2] = lr_data[0:new_H:2, 1:new_W:2] #B
                else:
                    data[1::2, 1::2, 0] = lr_data[1:new_H:2, 1:new_W:2] #R
                    data[1::2, 0::2, 1] = lr_data[1:new_H:2, 0:new_W:2] #G
                    data[0::2, 1::2, 1] = lr_data[0:new_H:2, 1:new_W:2] #G
                    data[0::2, 0::2, 2] = lr_data[0:new_H:2, 0:new_W:2] #B
                from utils.utils import matrix_multiplier
                data = matrix_multiplier(wb,data) 
                data = np.ascontiguousarray(data.transpose((2, 0, 1)))
                data_out = np.expand_dims(data,axis=0)
                data_out = torch.from_numpy(data_out).float()
                data_out = data_out.to(device, non_blocking=True)
                                        
                try:
                    #图像分辨率太大时，对原图进行裁剪，然后对输出的结果进行拼接
                    crop_output = []
                    #对测试图片裁剪，raw和isp图像对
                    crop_data, h, w, mask = utils.crop_single_test(data_out)
                    print('file pre-process done!')
                    print('starting inference······')                    
                    #对batch每一对计算
                    for i in range(len(crop_data)):
                        crop_model_out = model(crop_data[i])
                        #模型输出有多个时，选择第一个
                        crop_model_rgb = crop_model_out if len(crop_model_out) == 1 else crop_model_out[0]  
                        #数据限制范围0~1
                        crop_model_rgb = crop_model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                        #一个batch存为一个列表
                        crop_output.append(crop_model_rgb)
                    #model_out = model(data_out[:,:,0:2048,0:4096])
                    model_out = utils.cat_test(crop_output, h, w, mask)
                    model_rgb = model_out if len(model_out) == 1 else model_out[0]#模型输出有多个时，选择第一个
                    model_rgb = model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                    out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()

                    if args.b_save_results:
                        filename, _ = os.path.splitext(img_names[example_num])
                        path = os.path.join('./inference/result/', filename +'_'+args.s_model.split('.')[0]+ '.jpg')
                        print('saving output······')
                        cv2.imwrite(path, cv2.cvtColor(np.uint8(out_data * 255), cv2.COLOR_RGB2BGR))
                        print('--------------------------------------------------')

                except Exception as e:
                    utils.catch_exception(e)
    return print('All files done!')
def inference_srgb_input(model):
    #使用数据集外的数据进行推理，同理对输入的RAW图像进行简单的预处理，包括黑电平校正，初始白平衡
    #然后将3通道的RAW和ISP sRGB同时送入网络进行计算，将网络输出结果直接保存即可
    #推理时，直接运行 main_inference.py
    #导入的模型和test一样在config里进行设置实验文件夹以调用对应的best pth
    #值得注意的是RAW图像和ISP的图像尺寸关系，直接使用photoshop导出的尺寸略小于raw的尺寸
    #需要使用exiftoolGUI删除默认裁剪标签，确保尺寸问题
    #相机的isp图像一般小于rawpy读取的raw图像，可能源于裁剪标签，当然也可以读取裁剪标签对raw进行裁剪而不改动isp
    #如果遇到那种删不掉裁剪标签的相机，可以先使用dng conventor转换为标准的dng格式，再使用raw2dng重新封装为dng，然后再用Photoshop打开导出为jpg
    #这样有个问题就是jpg不一定符合原始相机的风格
    device = torch.device('cpu' if args.b_cpu else 'cuda')
    base_address = './inference/input_data/'
    from glob import glob
    img_names = [os.path.basename(x) for x in sorted(glob(os.path.join(base_address,'isp', '*.*')))]
    model.eval()
    print('All files names:',img_names)
    with torch.no_grad():
        for example_num in range(len(img_names)):
            print('processing %d/%d'%(example_num+1,len(img_names)) )
            import imageio as im
            print('file:%s is pre-processing...'%(img_names[example_num]))
            isp = im.imread(os.path.join(base_address,'isp', img_names[example_num]))
            print('srgb file shape:(%d,%d,%d)'%(isp.shape[0],isp.shape[1],isp.shape[2]))             
            isp_data = np.expand_dims(isp,axis=0)
            isp_data = np.ascontiguousarray(isp_data.transpose((0, 3, 1, 2)))
            isp_data = torch.from_numpy(isp_data).float()/255
            isp_data = isp_data.to(device, non_blocking=True)
                                    
            try:
                #图像分辨率太大时，对原图进行裁剪，然后对输出的结果进行拼接
                crop_output = []
                #对测试图片裁剪，raw和isp图像对
                crop_isp, h, w, mask = utils.crop_single_test(isp_data)
                print('file pre-process done!')
                print('starting inference······')
                #对batch每一对计算
                for i in range(len(crop_isp)):
                    crop_model_out = model(crop_isp[i])
                    #模型输出有多个时，选择第一个
                    crop_model_rgb = crop_model_out if len(crop_model_out) == 1 else crop_model_out[0]  
                    #数据限制范围0~1
                    crop_model_rgb = crop_model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                    #一个batch存为一个列表
                    crop_output.append(crop_model_rgb)
                #model_out = model(data_out[:,:,0:2048,0:4096])
                model_out = utils.cat_test(crop_output, h, w, mask)
                model_rgb = model_out if len(model_out) == 1 else model_out[0]#模型输出有多个时，选择第一个
                model_rgb = model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()
                if args.b_save_results:
                    filename, _ = os.path.splitext(img_names[example_num])
                    path = os.path.join('./inference/result/', filename +'_'+args.s_model.split('.')[0]+ '.jpg')
                    print('saving output······')
                    cv2.imwrite(path, cv2.cvtColor(np.uint8(out_data * 255), cv2.COLOR_RGB2BGR))
                    print('--------------------------------------------------')
            except Exception as e:
                utils.catch_exception(e)
    return print('All files done!')
