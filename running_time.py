import torch
import os
import numpy as np
from utils import utils
import argparse

parser = argparse.ArgumentParser(description='Running times Configure File')
parser.add_argument('--input_type', type=str, default='dual', help='the input type')#dual,single(raw or srgb)
parser.add_argument('--model_path', type=str, default='baseline', help='model root path')
parser.add_argument('--s_experiment_name', type=str, default='DarkSRv65v11', help='experiment name')
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0', help='which GPU to use')
parser.add_argument('--s_model', '-m', default=parser.parse_known_args()[0].s_experiment_name+'.'+\
                    parser.parse_known_args()[0].s_experiment_name, help='model name')
args = parser.parse_args(args=[])

os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
device = torch.device('cuda')
model = utils.import_fun(args.model_path, args.s_model.strip())()   
model = model.to(device)

if args.input_type == 'dual':
    data = torch.randn(1, 3, 256, 256, dtype=torch.float32).to(device, non_blocking=True)
    isp = torch.randn(1, 3, 256, 256, dtype=torch.float32).to(device, non_blocking=True)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    turns = 200
    timings = np.zeros((turns, 1))
    # start = perf_counter()
    # GPU-Warm-Up
    warm_up = 10
    with torch.no_grad():
        for _ in range(warm_up):  # 热身，使GPU利用率达到100%，更精确地计算推理时间
            full = model(data,isp)
        # Measure Performance
        for i in range(0, turns):
            starter.record()
            full = model(data,isp)
            ender.record()
            # wait for GPU sync
            torch.cuda.synchronize()  # 等待同步
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time
    mean_syn = np.sum(timings) / turns
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.2f}ms Std@5 {std_syn:.2f}ms FPS@1 {mean_fps:.2f}'.\
          format(mean_syn=mean_syn,std_syn=std_syn, mean_fps=mean_fps))
    print(mean_syn)
    
else:# for raw or srgb input model
    data = torch.randn(1, 3, 256, 256, dtype=torch.float32).to(device, non_blocking=True)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    turns = 200
    timings = np.zeros((turns, 1))
    # start = perf_counter()
    # GPU-Warm-Up
    warm_up = 10
    with torch.no_grad():
        for _ in range(warm_up):  # 热身，使GPU利用率达到100%，更精确地计算推理时间
            full = model(data)
        # Measure Performance
        for i in range(0, turns):
            starter.record()
            full = model(data)
            ender.record()
            # wait for GPU sync
            torch.cuda.synchronize()  # 等待同步
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time
    mean_syn = np.sum(timings) / turns
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.2f}ms Std@5 {std_syn:.2f}ms FPS@1 {mean_fps:.2f}'.\
          format(mean_syn=mean_syn,std_syn=std_syn, mean_fps=mean_fps))
    print(mean_syn)


