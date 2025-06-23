import torch
import os
from utils import utils
from thop import profile
import argparse

parser = argparse.ArgumentParser(description='Flops Configure File')
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
    isp = torch.randn((1, 3, 256, 256), dtype=torch.float32).to(device, non_blocking=True)
    flops, params = profile(model, inputs=(data,isp,))
    print('flops:%.2fG'%(flops/1e9))
else:
    data = torch.randn(1, 3, 256, 256, dtype=torch.float32).to(device, non_blocking=True)
    flops, params = profile(model, inputs=(data,))
    print('flops:%.2fG'%(flops/1e9))

pytorch_total_params = sum(p.numel() for p in model.parameters())
trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total:%.3fM, Trainable:%.3fM'%(pytorch_total_params/1e6,trainable_pytorch_total_params/1e6))
