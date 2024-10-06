import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
'''
use wavelet transform to be a framework

'''
 
# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def dwt(self,x):

        x01 = x[:, :, 0::2, :] / 2#4,3,128,256
        x02 = x[:, :, 1::2, :] / 2#4,3,128,256
        x1 = x01[:, :, :, 0::2]#4,3,128,128
        x2 = x02[:, :, :, 0::2]#4,3,128,128
        x3 = x01[:, :, :, 1::2]#4,3,128,128
        x4 = x02[:, :, :, 1::2]#4,3,128,128
        x_LL = x1 + x2 + x3 + x4 #4,3,128,128
        x_HL = -x1 - x2 + x3 + x4#4,3,128,128
        x_LH = -x1 + x2 - x3 + x4#4,3,128,128
        x_HH = x1 - x2 - x3 + x4#4,3,128,128

        return x_LL, x_HL, x_LH, x_HH

    def forward(self, x):
        return self.dwt(x)

# 逆向二维离散小波
class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def idwt(self,x):

        b,c,h,w = x[0].shape
        x1 = x[0][:,:,:,:]/2
        x2 = x[1][:,:,:,:]/2
        x3 = x[2][:,:,:,:]/2
        x4 = x[3][:,:,:,:]/2

        h = torch.zeros([b, c, h*2,w*2]).float().cuda()

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

    def forward(self, x):
        return self.idwt(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

class FeedForward(nn.Module):
    def __init__(self, num_dim, bias):
        super(FeedForward, self).__init__()

        self.encoder = nn.Conv2d(num_dim, num_dim, kernel_size=1, bias=bias)
        self.decoder = nn.Conv2d(num_dim, num_dim, kernel_size=1, bias=bias)
        self.conv1 = nn.Conv2d(num_dim,num_dim*2, kernel_size=1,bias=bias)
        self.scala = nn.Conv2d(num_dim,num_dim, kernel_size=1, bias=bias)
        self.shift = nn.Conv2d(num_dim,num_dim, kernel_size=1, bias=bias)

    def forward(self, guide,attened):
        x = self.encoder(guide)
        x1, x2 = self.conv1(attened).chunk(2, dim=1)
        alpha = self.scala(x1)
        beta = self.shift(x2)
        out = guide + self.decoder(x * alpha + beta)
        return out

class Attention(nn.Module):
    def __init__(self, num_dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv2d(num_dim, num_dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(num_dim * 2, num_dim * 2, kernel_size=3, 
                                   stride=1, padding=1, groups=num_dim * 2, bias=bias)
        self.q = nn.Conv2d(num_dim, num_dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(num_dim, num_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(num_dim, num_dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        #self.encoder_R = nn.Conv2d(3,dim,3,1,1)
        #self.encoder_S = nn.Conv2d(3,dim,3,1,1)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, bias)
        self.out3 = nn.Conv2d(dim,3,3,1,1)

    def forward(self, input_R, input_S):
        #input_R = self.encoder_R(input_R)#[3,32,32]
        input_R_normal = self.norm1(input_R)#[32,32,32]
        #input_S = self.encoder_S(input_S)
        input_S_normal = self.norm1(input_S)#[32,32,32]
        input_R_attened = input_R + self.attn(input_R_normal, input_S_normal)
        out = self.ffn(input_S,input_R_attened)
        #out1 = self.out3(out)
        return out

class ChannelAttentionLayer(nn.Module):
    # Channel Attention (CA) Layer
    # 通道注意力层，输出为 输入*通道注意力
    def __init__(self, channel, bias):
        super(ChannelAttentionLayer, self).__init__()
        # global average pooling: feature --> point，一个通道化为一个点
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#自适应平均池化，指定输出尺寸
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, 4, 1, padding=0, bias=bias),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(4, channel, 1, padding=0, bias=bias),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    #残差通道注意力块
    def __init__(self, in_channel):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(*[
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True),
            ChannelAttentionLayer(in_channel, True)
        ])
    def forward(self, x):
        # x+x*y，输出为 输入+乘以通道注意力后的输入
        out = self.body(x)
        out += x
        return out

class RCAG(nn.Module):
    #残差通道注意力组，RCAB的组合
    def __init__(self, num_RCAB, inchannel):
        super(RCAG, self).__init__()
        body = []
        for i in range(num_RCAB):
            body.append(RCAB(in_channel=inchannel))
        body.append(nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=(3 - 1) // 2, stride=1))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        out = self.body(x)
        out += x
        #输入加上 n个RCAB级联后经过3*3卷积
        return out
### DEM
class Wavelet_NET(nn.Module):
    def __init__(self,dim_in=32,dim_out=3,num_FAM=1,num_RCAG=3,num_RCAB=4):
        super().__init__()

        self.num_FAM = num_FAM
        self.num_RCAG = num_RCAG

        if num_FAM == 1:
            self.FAM = TransformerBlock(dim_in)
        else :
            self.FAM1 = TransformerBlock(dim_in)
            self.FAM2 = TransformerBlock(dim_in)
            
        self.RCAGs = nn.ModuleList()
        for i in range(self.num_RCAG):
            self.RCAGs.append(
                RCAG(num_RCAB, inchannel=dim_in))
            
        self.GFF = nn.Sequential(*[
            nn.Conv2d((self.num_RCAG+1) * dim_in, dim_in, 1, padding=0, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim_in, dim_in, 3, padding=(3-1)//2, stride=1)])

        self.output = nn.Sequential(*[
            nn.Conv2d(dim_in, dim_out * 2 * 2, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(2)])

    def forward(self, raw, rgb):
        if self.num_FAM == 1:
            feat = self.FAM(raw,rgb)
            feat = raw + feat
        else:
            feat = self.FAM1(raw,rgb)
            feat1 = raw +feat
            feat2 = self.FAM2(feat1,rgb)
            feat = raw + feat2

        x = feat
        RCAGs_out = [x]
        for i in range(self.num_RCAG):
            x = self.RCAGs[i](x)
            RCAGs_out.append(x)
        #print('raw feature:',raw.shape)
        x = self.GFF(torch.cat(RCAGs_out,1))
        x = x + feat
        out = self.output(x)        
        return out

class TransformerBlock_self(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock_self, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, bias)

    def forward(self, input_R):
        # input_ch = input_R.size()[1]
        input_R1 = self.norm1(input_R)
        input_R2 = self.norm2(input_R)
        input_R_attened = input_R + self.attn(input_R1, input_R2)
        out = self.ffn(input_R,input_R_attened)
        return out
 
class RAW(nn.Module):
    def __init__(self,dim_in=32,dim_out=32,num_RCAG=2,num_RCAB=4):
        super().__init__()
        self.num_RCAG = num_RCAG
        self.encoder = nn.Sequential(nn.Conv2d(4,dim_in,3,1,1))
        self.RCAGs = nn.ModuleList()
        for i in range(self.num_RCAG):
            self.RCAGs.append(
                RCAG(num_RCAB, inchannel=dim_in)
            )
        self.GFF = nn.Sequential(*[
            nn.Conv2d((self.num_RCAG+1) * dim_in, dim_in, 1, padding=0, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim_in, dim_in, 3, padding=(3-1)//2, stride=1)
        ])

        self.output = nn.Sequential(*[
            nn.Conv2d(dim_in, dim_out * 2 * 2, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(2)])

    def forward(self, raw):
        feat  =self.encoder(raw)
        x = feat
        RCAGs_out = [x]
        for i in range(self.num_RCAG):
            x = self.RCAGs[i](x)
            RCAGs_out.append(x)
        x = self.GFF(torch.cat(RCAGs_out,1))
        x = x + feat
        feat = self.output(x)
        return feat

class RGB(nn.Module):
    def __init__(self,dim_in=32,dim_out=32,num_RCAG=2,num_RCAB=4):
        super().__init__()
        self.num_RCAG = num_RCAG
        self.encoder = nn.Sequential(nn.Conv2d(3,dim_in,3,1,1))
        self.RCAGs = nn.ModuleList()
        for i in range(num_RCAG):
            self.RCAGs.append(
                RCAG(num_RCAB, inchannel=dim_in)
            )
        self.GFF = nn.Sequential(*[
            nn.Conv2d((self.num_RCAG+1) * dim_in, dim_in, 1, padding=0, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim_in, dim_in, 3, padding=(3-1)//2, stride=1)
        ])
        self.output = nn.Sequential(*[
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1, bias=True)])

    def forward(self, raw):
        feat  =self.encoder(raw)
        x = feat
        RCAGs_out = [x]
        for i in range(self.num_RCAG):
            x = self.RCAGs[i](x)
            RCAGs_out.append(x)
        x = self.GFF(torch.cat(RCAGs_out,1))
        x = x + feat
        feat = self.output(x)
        return feat

class JSLNet(nn.Module):
    def __init__(self,
                 num_wavelet=4, # LL,LH,HL,HH
                 pre_in_dim=36,
                 pre_out_dim=36,
                 post_in_dim=36,
                 post_out_dim=3, # RGB output
                 num_RCAG=2,
                 num_FAM=1,# 1 or 2 
                 num_RCAB=4):
        super().__init__()

        self.num_wavelet = num_wavelet
        self.rgb = RGB(pre_in_dim,pre_out_dim,num_RCAG,num_RCAB)
        self.raw = RAW(pre_in_dim,pre_out_dim,num_RCAG,num_RCAB)
        self.raw_dwt_decom = DWT()#[256,256,dim_out]-[256,256,dim_out]x4
        self.rgb_dwt_decom = DWT()#[256,256,dim_out]-[256,256,dim_out]x4
        self.idwt_recons = IDWT()#[256,256,3]x4->[512,512,3]
        self.conv_last = nn.Conv2d(3,3,3,1,1)

        for i in range(self.num_wavelet):
            self.__setattr__('wavelet_layer_{}'.format(i),\
            Wavelet_NET(post_in_dim,post_out_dim,num_FAM,num_RCAG,num_RCAB))
            
    def rgb2raw(self,img):       # pack [H,W,3] -> [H//2,W//2,4]
        b,_,h,w = img.shape
        raw = torch.zeros([b,4,h//2,w//2],dtype=torch.float).cuda()
        img = img.squeeze(1)
        raw[:, 0, :, :] = img[:, 0, ::2, ::2]
        raw[:, 1, :, :] = img[:, 1, ::2, 1::2]
        raw[:, 2, :, :] = img[:, 1, 1::2, ::2]
        raw[:, 3, :, :] = img[:, 2, 1::2, 1::2]
        return raw

    def forward(self, raw, isp):
        rgb_feat = self.rgb_dwt_decom(self.rgb(isp))
        #print('isp feature:',rgb_feat[0].shape)
        raw_pack = self.rgb2raw(raw)
        raw_feat = self.raw_dwt_decom(self.raw(raw_pack))
        trans_wavelets = []
        for i in range(self.num_wavelet):
            trans_wavelet = self.__getattr__('wavelet_layer_{}'.format(i))\
                                                 (raw_feat[i],rgb_feat[i])
            trans_wavelets.append(trans_wavelet)
        out = self.idwt_recons(trans_wavelets)
        out = self.conv_last(out)
        return out

if __name__=='__main__':
    from thop import profile
    model = JSLNet().cuda()
    a = torch.randn((1,3,256,256)).cuda()
    b = torch.randn((1,3,256,256)).cuda()

    c = model(a,b)
    print(c.shape)
    flops, params = profile(model, inputs=(a,b))
    print('flops:',flops,'params:',params)
    print('flops:%.2fG, params:%.3fM'%(flops/1e9,params/1e6))

    from pytorch_model_summary import summary

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Total - ', pytorch_total_params)
    print('Trainable - ', trainable_pytorch_total_params)

