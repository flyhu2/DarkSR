
import os
import time
import random
import numpy as np
import math
import torch
import torch.nn.parallel.data_parallel
import torch.backends.cudnn
import torch.optim.lr_scheduler
import torch.distributed
import torch.multiprocessing
from torch.utils.data import DataLoader

import main_function as main_function
from config.config import args
from utils import log, timer, check_point, utils
from utils.utils import creat_info_csv,add_to_csv,creat_metr_csv,save_to_csv


def main():
    torch.cuda.empty_cache()
    random.seed(args.n_seed)
    np.random.seed(args.n_seed)
    torch.manual_seed(args.n_seed)
    torch.backends.cudnn.enabled = args.b_cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    device = torch.device('cpu' if args.b_cpu else 'cuda')

    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    info_log.write('Experiment: {} ({})'.format(
        args.s_experiment_name,
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    ))

    if args.model_need_args == True:
        model = utils.import_fun(args.model_path, args.s_model.strip())(args) # 需要传入参数时候填入args，不需要的模型则去掉
    else :
        model = utils.import_fun(args.model_path, args.s_model.strip())()   
    #model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpu))) if args.n_gpu > 1 and not args.b_cpu else model
    model = model.to(device)
    
    pre_model = utils.import_fun('models', args.pre_model.strip())(args)
    pre_model = pre_model.to(device)

    if args.use_ema:
        model_ema = utils.ModelEMA(model)
    else:
        model_ema = None

    total_params = utils.calc_para(model)
    info_log.write('Total number of parameters: {}'.format(total_params))
    
    pre_model_params = utils.calc_para(pre_model)
    info_log.write('Total number of pre-model parameters: {}'.format(pre_model_params))

    loss = utils.import_fun('loss', args.s_loss_model.strip())(args).to(device)
    ckp = check_point.CheckPoint('./experiments', args.s_experiment_name)

    global data_train
    data_train = utils.import_fun('dataload', args.s_train_dataset.strip())(args, b_train=True)
    train_sampler = None
    data_loader_train = DataLoader(
        dataset=data_train,
        num_workers=args.n_threads,
        batch_size=args.n_batch_size,
        shuffle=(train_sampler is None),
        pin_memory=not args.b_cpu,
        sampler=train_sampler)
    data_loader_test = [(DataLoader(
                            dataset=utils.import_fun('dataload', dataset)(args, b_train=False),
                            num_workers=0,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=not args.b_cpu
                        )) for dataset in args.s_eval_dataset.strip().split('+')]

    optimizer = None
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     betas=args.betas,
                                     eps=args.epsilon,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.lr,
                                      betas=args.betas,
                                      eps=args.epsilon,
                                      weight_decay=args.weight_decay)


    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    lr_init = optimizer.param_groups[0]['lr']

    load_epoch = 0
    if args.b_resume or args.b_test_only:
        pth = ckp.load(device, args.b_load_best)
        model.load_state_dict(pth.get('model'))
        #model.load_state_dict(torch.load('model_best.pth'))
        loss.load_state_dict(pth.get('loss'))
        #optimizer.load_state_dict(pth.get('optimizer'))
        #scheduler.load_state_dict(pth.get('scheduler'))
        load_epoch = pth.get('epoch')

    ckp.save_config(args, model)

    timer_all = timer.Timer(True)
    timer_epoch = timer.Timer()

    if args.b_test_only:
        #创建保存指标的csv文件
        creat_metr_csv()
        info_log.write('Resume model for testing')
        for ds in data_loader_test:
            info_log.write('Testing database: {}({})'.format(ds.dataset.name, len(ds)))

        if args.input_type == 'dual':
            psnr, ssim, deltae, test_time = main_function.dual_test(model, data_loader_test)
        elif args.input_type == 'raw':
            psnr, ssim, deltae, test_time = main_function.raw_test(model, data_loader_test)
        elif args.input_type == 'srgb':
            psnr, ssim, deltae, test_time = main_function.srgb_test(model, data_loader_test)

        info_log.write('[Testing: {:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}, {:.5f}]'.format(
            psnr[:, -1].item(),
            psnr[:, 0].item(), 
            psnr[:, 1].item(), 
            psnr[:, 2].item(),
            ssim[-1].item(),
            deltae[-1].item(),
        ))
        save_test = ['Testing'.ljust(10),
                    '{:.3f}'.format(psnr[:, -1].item()).ljust(10),
                    '{:.3f}'.format(psnr[:, 0].item()).ljust(10),
                    '{:.3f}'.format(psnr[:, 1].item()).ljust(10),
                    '{:.3f}'.format(psnr[:, 2].item()).ljust(10),
                    '{:.5f}'.format(ssim.item()).ljust(10),
                    '{:.5f}'.format(deltae.item()).ljust(10)]
        save_to_csv(save_test)
        info_log.write('Testing elapsed: {:.3f}s'.format(test_time))
        save_to_csv(['Elapsed'.ljust(10),'{:.3f}s'.format(test_time).ljust(10)])

    else:
        #创建保存训练时log信息的csv文件
        creat_info_csv()
        str_loss = 'Loss function: '
        for l in loss.get_loss():
            if l.get('function') is not None:
                str_loss = str_loss + '[{:.3f} * {}]'.format(l.get('weight'), l.get('type'))
        info_log.write(str_loss)

        if args.b_resume:
            info_log.write('Resume model from epoch {} for training'.format(load_epoch + 1))

        if args.joint:
            pre_pth = ckp.load_pre_model(device)  # 加载预训练模型结构
            pretrained_dict = pre_pth.get('model')
            model_dict = model.state_dict()  # 完整模型的参数
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        info_log.write('Training database: {}({})'.format(data_train.name, len(data_train)))
        for ds in data_loader_test:
            info_log.write('Testing database: {}({})'.format(ds.dataset.name, len(ds)))

        best_cpsnr = {'psnr': torch.zeros(3, device=device, requires_grad=False), 'cpsnr': torch.zeros(1, device=device, requires_grad=False), 'epoch': 0}
        for epoch in range(load_epoch, args.n_epochs):

            add_to_csv(0,'[{}/{}]'.format(epoch + 1,args.n_epochs))
            info_log.write('\n[Epoch: {}/{}]'.format(epoch + 1,args.n_epochs,))

            timer_epoch.start()
            # pth = ckp.load_restoration(device, args.b_load_best)  # 加载细节支路的预训练模型的路径
            # Restoration_model.load_state_dict(pth.get('model'))  # 加载细节支路的预训练模型
            # losses = common.train(Restoration_model, Color_model, data_loader_train, loss, optimizer)
            
            if args.input_type == 'dual':
                losses = main_function.dual_train(model, data_loader_train, loss, optimizer, model_ema)
            elif args.input_type == 'raw':
                losses = main_function.raw_train(model, data_loader_train, loss, optimizer, model_ema)
            elif args.input_type == 'srgb':
                losses = main_function.srgb_train(model, data_loader_train, loss, optimizer, model_ema)
            
            # scheduler.step(None)
            test_time = 0
            if (epoch + 1) % args.n_epochs_per_evaluation == 0:
                if args.input_type == 'dual':
                    psnr, ssim,deltae ,test_time = main_function.dual_test(model, data_loader_test)
                elif args.input_type == 'raw':  
                    psnr, ssim, deltae, test_time = main_function.raw_test(model, data_loader_test)
                elif args.input_type == 'srgb':
                    psnr, ssim, deltae, test_time = main_function.srgb_test(model, data_loader_test)
                # 保存cpsnr最好的epoch
                if psnr[:, -1] > best_cpsnr.get('cpsnr'):
                    best_cpsnr['cpsnr'] = psnr[0, -1]
                    best_cpsnr['psnr'] = psnr[0, :3]
                    best_cpsnr['ssim'] = ssim[-1]
                    best_cpsnr['deltae'] = deltae[-1]
                    best_cpsnr['epoch'] = epoch + 1
                add_to_csv(2,'[Iter: {}/{}] [CPSNR: {:.3f}([R:{:.3f}] [G: {:.3f}] [B: {:.3f}])] [SSIM: {:.5f}] [DeltaE: {:.5f}]'.format(
                            epoch + 1, 
                            args.n_epochs,
                            psnr[:, -1].item(),
                            psnr[:, 0].item(), 
                            psnr[:, 1].item(), 
                            psnr[:, 2].item(),
                            ssim[-1].item(),
                            deltae[-1].item()))

                info_log.write('[Epoch: {}/{}, {:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}, {:.5f}]'.format(
                    epoch + 1, args.n_epochs,
                    psnr[:, -1].item(),
                    psnr[:, 0].item(), psnr[:, 1].item(), psnr[:, 2].item(),
                    ssim[-1].item(),
                    deltae[-1].item()
                ))
                add_to_csv(3,'[CPSNR: {:.3f}([R:{:.3f}] [G: {:.3f}] [B: {:.3f}])][SSIM: {:.5f}] [DeltaE: {:.5f}] [@epoch: {}]'.format(
                            best_cpsnr.get('cpsnr').item(),
                            best_cpsnr.get('psnr')[0].item(), best_cpsnr.get('psnr')[1].item(), best_cpsnr.get('psnr')[2].item(),
                            best_cpsnr.get('ssim').item(),
                            best_cpsnr.get('deltae').item(),
                            best_cpsnr.get('epoch')
                            ))
                info_log.write('[Best CPSNR: {:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f} [DeltaE: {:.5f}] [@epoch {}]'.format(
                    best_cpsnr.get('cpsnr').item(),
                    best_cpsnr.get('psnr')[0].item(), best_cpsnr.get('psnr')[1].item(), best_cpsnr.get('psnr')[2].item(),
                    best_cpsnr.get('ssim').item(),
                    best_cpsnr.get('deltae').item(),
                    best_cpsnr.get('epoch')
                ))
                if (epoch + 1) % args.n_epochs_per_save == 0:
                    ckp.save(model.module if args.n_gpu > 1 else model, loss, losses, optimizer, epoch + 1,
                             is_best=(best_cpsnr.get('epoch') == epoch + 1), result=psnr.cpu())
            timer_epoch.stop()
            info_log.write('[Epoch elapsed: {:.1f}s/{:.1f}s]'.format(test_time, timer_epoch.elapsed_ticks()))
            add_to_csv(4,'[Epoch time: {:.1f}s/{:.1f}s]'.format(test_time, timer_epoch.elapsed_ticks()))

            all_time = timer.elapsed_ticks_format(timer_all.elapsed_ticks())
            left_time = timer.elapsed_ticks_format(
                (args.n_epochs - (epoch + 1)) * (timer_all.elapsed_ticks() / (epoch + 1 - load_epoch)))
            info_log.write('[All elapsed: {:d}h{:d}m{:d}s. Left time: {:d}h{:d}m{:d}s]'.format(
                all_time[0], all_time[1], all_time[2], left_time[0], left_time[1], left_time[2],
            ))
            add_to_csv(5,'[Time elapsed: {:d}h{:d}m{:d}s]. [Left time: {:d}h{:d}m{:d}s]'.format(
                all_time[0], all_time[1], all_time[2], left_time[0], left_time[1], left_time[2]))
            
            if os.path.exists(os.path.join('./experiments', args.s_experiment_name, 'cancel.key')):
                os.remove(os.path.join('./experiments', args.s_experiment_name, 'cancel.key'))
                info_log.write('Cancel !!!')
                return
    info_log.write('Completed !!!')