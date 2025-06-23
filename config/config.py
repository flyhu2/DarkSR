import argparse
parser = argparse.ArgumentParser(description='DarkSR Configure File')
'''
------------------------2号机主要进行raw方法实验，使用方法：-------------------------
1. 修改input_type为指定类型，双输入，单输入
2. 设置实验名，一般时间+模型
3. 是否测试，这里训练时不进行图片保存，在实验跑完后，在设置仅测试来输出最佳模型的输出图片
4. 测试时是否裁剪，一方面可以减少显存占用，另一方面可以避免尺寸问题
5. 2号机器3张都是3090
6. epo默认500，每10个epo评估一次，patch大小256，batch大小6
----------------------------------------------------------------------------------
'''
parser.add_argument('--input_type', type=str, default='srgb', help='the input type')# dual,raw,srgb
parser.add_argument('--camera_model', type=str, default='D700_patch', help='the input type')# 5D,20D,450D,D700,All
parser.add_argument('--s_experiment_name', type=str, default='1002_D700_RetinexSRFormer', help='experiment name')
parser.add_argument('--b_test_only', type=bool, default=False, help='set to test the model')
parser.add_argument('--b_resume', type=bool, default=False,  help='resume from specific checkpoint')#
parser.add_argument('--b_crop_when_test', type=bool, default=True, help='if crop when testing to save GPU')#
parser.add_argument('--n_crop_size', type=int, default=256, help='crop test image')#

parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='1', help='which GPU to use')
parser.add_argument('--model_path', type=str, default='supplementary', help='model root path')#baseline文件夹下的模型

#------------SRFormer.SRFormer, SID.SIDUNet, DarkSR.DarkSR ,TCI.TCI,Master_base.Restoration-------------
parser.add_argument('--model_need_args', type=bool, default=False, help='model path')#是否需要传递一些参数到模型
parser.add_argument('--s_model', '-m', default=parser.parse_known_args()[0].s_experiment_name.split('_')[-1]+'.'+\
                        parser.parse_known_args()[0].s_experiment_name.split('_')[-1], help='model name')
parser.add_argument('--n_patch_size', type=int, default=128, help='patch size')# even number，cause RGGB bayer pattern
parser.add_argument('--n_batch_size', type=int, default=6, help='batch size for training')#12

#--------------------------------Epoch和多少个Epoch测试一次模型-----------------------------------
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs to train')#100 300
parser.add_argument('--n_epochs_per_evaluation', type=int, default=10, help='how many evaluation model')

#-----------------------------训练和测试范围，测试代码能否运行时选择小一点的范围--------------------------------------
#5D：29/283,20D：17/170,450D：19/252，D700:17/220
if parser.parse_known_args()[0].camera_model == '5D':
    parser.add_argument('--train_range', type=str, default='1-283/0-0', help='')
    parser.add_argument('--test_range', type=str, default='0-0/1-29', help='')
elif parser.parse_known_args()[0].camera_model == '20D_patch':
    parser.add_argument('--train_range', type=str, default='1-680/0-0', help='')
    parser.add_argument('--test_range', type=str, default='0-0/1-68', help='')
elif parser.parse_known_args()[0].camera_model == '450D':
    parser.add_argument('--train_range', type=str, default='1-252/0-0', help='')
    parser.add_argument('--test_range', type=str, default='0-0/1-19', help='')
elif parser.parse_known_args()[0].camera_model == 'D700_patch':
    parser.add_argument('--train_range', type=str, default='1-880/0-0', help='')
    parser.add_argument('--test_range', type=str, default='0-0/1-68', help='')
elif parser.parse_known_args()[0].camera_model == 'All':
    parser.add_argument('--train_range', type=str, default='1-1395/0-0', help='')
    parser.add_argument('--test_range', type=str, default='0-0/1-153', help='')
parser.add_argument('--n_epochs_per_save', type=int, default=1, help='how many batches to ')
#----------------------训练时不保存测试图片，节省时间，选择不同的输入时选择不同的dataload------------------------------
if parser.parse_known_args()[0].b_test_only:
    parser.add_argument('--b_save_results', type=bool, default=True, help='save output results')# , save output image
else:
    parser.add_argument('--b_save_results', type=bool, default=False, help='save output results')# ,when train not save output image
if parser.parse_known_args()[0].input_type == 'dual':
    parser.add_argument('--s_train_dataset', '-t', default='DarkSR_train.DarkSR_train', help='training data model name')
    parser.add_argument('--s_eval_dataset', '-e', default='DarkSR_test.DarkSR_test', help='evaluation dataset')
elif parser.parse_known_args()[0].input_type == 'raw':
    parser.add_argument('--s_train_dataset', '-t', default='RAW_train.RAW_train', help='training data model name')
    parser.add_argument('--s_eval_dataset', '-e', default='RAW_test.RAW_test', help='evaluation dataset')
elif parser.parse_known_args()[0].input_type == 'srgb':
    parser.add_argument('--s_train_dataset', '-t', default='SRGB_train.SRGB_train', help='training data model name')
    parser.add_argument('--s_eval_dataset', '-e', default='SRGB_test.SRGB_test', help='evaluation dataset')

#------------------------------------统一使用L1 Loss-----------------------------------------
parser.add_argument('--s_loss_model', '-l', default='loss_base.LossBase', help='loss model name')# loss_base.LossBase
parser.add_argument('--s_loss', type=str, default='1*L1', help='loss function configuration')

parser.add_argument('--n_random_train', type=int, default=500, help='number of random training data blocks')

# Hardware specifications,硬件相关，包括GPU，多线程读取数据，随机种子等
parser.add_argument('--n_threads', type=int, default=0, help='number of threads for data loading')
parser.add_argument('--b_cpu', type=bool, default=False, help='use cpu only')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPU')
parser.add_argument('--b_cudnn', type=bool, default=False, help='use cudnn')
parser.add_argument('--n_seed', type=int, default=42, help='random seed')
parser.add_argument('--use_ema', type=bool, default=False, help='use modelEMA')

# Model specifications,模型相关，测试时导入的模型
parser.add_argument('--b_save_all_models', default=False, help='save all intermediate models')
parser.add_argument('--b_load_best', type=bool, default=True, help='use best model for testing')

# Data specifications,数据相关，数据集地址，裁剪块的大小，等
# Mixed All, Canon EOS 5D, Canon EOS 20D, Canon EOS 450D, Nikon D700
if parser.parse_known_args()[0].camera_model == '5D':
    parser.add_argument('--dir_dataset', type=str, default='../datasets/DarkSR/Canon EOS 5D', help='dataset directory')
elif parser.parse_known_args()[0].camera_model == '20D_patch':
    parser.add_argument('--dir_dataset', type=str, default='../datasets/DarkSR/20D_patch', help='dataset directory')
elif parser.parse_known_args()[0].camera_model == '450D':
    parser.add_argument('--dir_dataset', type=str, default='../datasets/DarkSR/Canon EOS 450D', help='dataset directory')
elif parser.parse_known_args()[0].camera_model == 'D700_patch':
    parser.add_argument('--dir_dataset', type=str, default='../datasets/DarkSR/D700_patch', help='dataset directory')
elif parser.parse_known_args()[0].camera_model == 'All':
    parser.add_argument('--dir_dataset', type=str, default='../datasets/DarkSR/Mixed All', help='dataset directory')
parser.add_argument('--scale', type=int, default=2, help='output scale size')
parser.add_argument('--n_rgb_range', type=int, default=1,  help='maximum value of RGB')
parser.add_argument('--data_pack', type=str, default='packet/packet', choices=('packet', 'bin', 'ori'), help='make binary data')
parser.add_argument('--b_make_reset', type=bool, default=False, help='re-make dataset')

# Optimization specifications，优化器相关
parser.add_argument('--lr', type=float, default=1.5e-4, help='learning rate')
parser.add_argument('--lrdecay_how_epos', type=int, default=5, help='learning rate')#defalut=5
parser.add_argument('--decay', type=str, default='20,40,60,80', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='AdamW', choices=('SGD', 'ADAM', 'AdamW'), help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')# 0.05 0!!!!!!!!!
parser.add_argument('--norm_type', type=float, default=2.0, help='gradient clipping norm_type (1.0 = L1, 2.0 = L2)')
parser.add_argument('--gclip', type=float, default=0, help='gradient clipping threshold (0 = no clipping)')

# Log specifications，日志记录
parser.add_argument('--n_batches_per_print', type=int, default=640, help='how many batches to wait before logging training status')   # 786

# Pre + Joint
parser.add_argument('--joint', type=bool, default=False,  help='joint from pr-train checkpoint')# True, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
parser.add_argument('--pretrain_dm_models', type=str, default='1018_stage1_2040', help='pre-trained model directory')
parser.add_argument('--pre_model', default='stage1.Restoration', help='pre-model name')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64, help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3, help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='A', help='parameters config of RDN. (Use in RDN)')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')

args = parser.parse_args(args=[])

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False






