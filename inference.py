import os
import torch
import main_function as main_function
from utils import check_point, utils
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='inference Configure File')
parser.add_argument('--input_type', type=str, default='dual', help='the input type')# dual,raw,srgb
parser.add_argument('--model_path', type=str, default='models', help='model root path')#baseline文件夹下的模型transformer
parser.add_argument('--s_experiment_name', type=str, default='0527_DarkSR_JSLNet', help='model name')
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='2', help='which GPU to use')
parser.add_argument('--s_model', '-m', default=parser.parse_known_args()[0].s_experiment_name.split('_')[2]+'.'+\
                    parser.parse_known_args()[0].s_experiment_name.split('_')[2], help='model.model')
parser.add_argument('--crop_when_inference', type=bool, default=True, help='crop test image')#
parser.add_argument('--n_crop_size', type=int, default=512, help='crop test image')#

args = parser.parse_args(args=[])

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
                rgb_data = im.imread_v2(os.path.join(base_address,'isp', img_names[example_num][:-4]+'.jpg'))
                #isp = cv2.resize(isp,(raw_data.shape[1],raw_data.shape[0]))
                #isp = raw.postprocess(no_auto_bright=True,user_flip=0)
                import numpy
                print('raw file shape:(%d,%d), srgb file shape:(%d,%d,%d)'%
                      (raw_data.shape[0],raw_data.shape[1],
                       rgb_data.shape[0],rgb_data.shape[1],rgb_data.shape[2]))                
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
                lr_rggb = numpy.zeros((lr_data.shape[0]-2,lr_data.shape[1]-2), dtype=float)
                lr_rggb = lr_data[1:-1, 1:-1] #R
                isp = rgb_data[1:-1,1:-1,:]
                print('raw shape after crop:(%d,%d),srgb shape after crop:(%d,%d,%d)'%
                      (lr_rggb.shape[0],lr_rggb.shape[1],
                       isp.shape[0],isp.shape[1],isp.shape[2]))                 
                new_H,new_W = lr_rggb.shape[0]//2*2, lr_rggb.shape[1]//2*2
                data = numpy.zeros((new_H,new_W,3), dtype=float)
                data[0::2, 0::2, 0] = lr_rggb[0:new_H:2, 0:new_W:2] #R
                data[1::2, 0::2, 1] = lr_rggb[1:new_H:2, 0:new_W:2] #G
                data[0::2, 1::2, 1] = lr_rggb[0:new_H:2, 1:new_W:2] #G
                data[1::2, 1::2, 2] = lr_rggb[1:new_H:2, 1:new_W:2] #B
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
                    if args.crop_when_inference:
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
                            crop_model_rgb = crop_model_rgb.mul(1.0).clamp(0, 1)
                            #一个batch存为一个列表
                            crop_output.append(crop_model_rgb)
                        model_out = utils.cat_test(crop_output, h, w, mask)
                    else:
                        model_out = model(data_out, isp_data)
                    model_rgb = model_out if len(model_out) == 1 else model_out[0]#模型输出有多个时，选择第一个
                    model_rgb = model_rgb.mul(1.0).clamp(0, 1)
                    out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()

                    filename, _ = os.path.splitext(img_names[example_num])
                    path = os.path.join('./inference/result/', filename +'_'+args.s_model.split('.')[0]+ '.jpg')
                    print('saving output······')
                    cv2.imwrite(path, cv2.cvtColor(np.uint8(out_data * 255), cv2.COLOR_RGB2BGR))
                    print('--------------------------------------------------')
                except Exception as e:
                    utils.catch_exception(e)
    return print('All files done!')
def inference_raw_input(model):
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
                lr_rggb = numpy.zeros((lr_data.shape[0]-2,lr_data.shape[1]-2), dtype=float)
                lr_rggb = lr_data[1:-1, 1:-1] #R
                print('raw file shape after crop:(%d,%d)'%(lr_rggb.shape[0],lr_rggb.shape[1]))                 
                new_H,new_W = lr_rggb.shape[0]//2*2, lr_rggb.shape[1]//2*2
                data = numpy.zeros((new_H,new_W,3), dtype=float)
                data[0::2, 0::2, 0] = lr_rggb[0:new_H:2, 0:new_W:2] #R
                data[1::2, 0::2, 1] = lr_rggb[1:new_H:2, 0:new_W:2] #G
                data[0::2, 1::2, 1] = lr_rggb[0:new_H:2, 1:new_W:2] #G
                data[1::2, 1::2, 2] = lr_rggb[1:new_H:2, 1:new_W:2] #B
                from utils.utils import matrix_multiplier
                data = matrix_multiplier(wb,data) 
                data = np.ascontiguousarray(data.transpose((2, 0, 1)))
                data_out = np.expand_dims(data,axis=0)
                data_out = torch.from_numpy(data_out).float()
                data_out = data_out.to(device, non_blocking=True)
                                        
                try:
                    if args.crop_when_inference:
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
                            crop_model_rgb = crop_model_rgb.mul(1.0).clamp(0, 1)
                            #一个batch存为一个列表
                            crop_output.append(crop_model_rgb)
                        model_out = utils.cat_test(crop_output, h, w, mask)
                    else:
                        model_out = model(data_out)
                    model_rgb = model_out if len(model_out) == 1 else model_out[0]#模型输出有多个时，选择第一个
                    model_rgb = model_rgb.mul(1.0).clamp(0, 1)
                    out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()

                    filename, _ = os.path.splitext(img_names[example_num])
                    path = os.path.join('./inference/result/', filename +'_'+args.s_model.split('.')[0]+ '.jpg')
                    print('saving output······')
                    cv2.imwrite(path, cv2.cvtColor(np.uint8(out_data * 255), cv2.COLOR_RGB2BGR))
                    print('--------------------------------------------------')

                except Exception as e:
                    utils.catch_exception(e)
    return print('All files done!')
def inference_srgb_input(model):
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
            isp = im.imread_v2(os.path.join(base_address,'isp', img_names[example_num]))
            print('srgb file shape:(%d,%d,%d)'%(isp.shape[0],isp.shape[1],isp.shape[2]))   
            isp_data = isp[1:-1,1:-1,:]   
            print('srgb shape after crop:(%d,%d,%d)'%(isp_data.shape[0],isp_data.shape[1],isp_data.shape[2]))       
            isp_data = np.expand_dims(isp_data,axis=0)
            isp_data = np.ascontiguousarray(isp_data.transpose((0, 3, 1, 2)))
            isp_data = torch.from_numpy(isp_data).float()/255
            isp_data = isp_data.to(device, non_blocking=True)
                                    
            try:
                if args.crop_when_inference:
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
                        crop_model_rgb = crop_model_rgb.mul(1.0).clamp(0, 1)
                        #一个batch存为一个列表
                        crop_output.append(crop_model_rgb)
                    model_out = utils.cat_test(crop_output, h, w, mask)
                else:
                    model_out = model(isp_data)

                model_rgb = model_out if len(model_out) == 1 else model_out[0]#模型输出有多个时，选择第一个
                model_rgb = model_rgb.mul(1.0).clamp(0, 1)
                out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()
                filename, _ = os.path.splitext(img_names[example_num])
                path = os.path.join('./inference/result/', filename +'_'+args.s_model.split('.')[0]+ '.jpg')
                print('saving output······')
                cv2.imwrite(path, cv2.cvtColor(np.uint8(out_data * 255), cv2.COLOR_RGB2BGR))
                print('--------------------------------------------------')
            except Exception as e:
                utils.catch_exception(e)
    return print('All files done!')

torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
device = torch.device('cuda')
model = utils.import_fun(args.model_path, args.s_model.strip())()   
model = model.to(device)
ckp = check_point.CheckPoint('./experiments', args.s_experiment_name)
pth = ckp.load(device, True)
model.load_state_dict(pth.get('model'))


if __name__ == '__main__':

    if args.input_type == 'dual':
        inference_dual_input(model)
    elif args.input_type == 'raw':
        inference_raw_input(model)
    elif args.input_type == 'srgb':
        inference_srgb_input(model)


