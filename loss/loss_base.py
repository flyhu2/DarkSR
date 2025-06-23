import torch
import torch.nn as nn
from config.config import args
from loss import ssim_loss


class LossBase(nn.Module):
    def __init__(self, args):
        super(LossBase, self).__init__()

        self.args = args
        self.device = torch.device('cpu' if args.b_cpu else 'cuda')

        self.loss = []

        for loss in self.args.s_loss.strip().split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'color':
                loss_function = nn.CosineSimilarity(dim=1, eps=1e-6)
            elif loss_type == 'VGG':
                loss_function = nn.L1Loss()
            elif loss_type == 'SSIM':
                loss_function = ssim_loss.SSIM()
            else:
                loss_function = None

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function,
                'value': 0.0
            })

        self.loss.append({'type': 'Total', 'weight': 0.0, 'function': None, 'value': 0.0})
    def get_loss(self):
        return self.loss
    def forward(self, model_out, target):
        losses = torch.zeros(len(self.loss), device=self.device)
        loss_sum = torch.zeros(1, device=self.device)
        for i, l in enumerate(self.loss):
            if l.get('function') is not None:
                if l.get('type') == 'color':
                    loss = torch.mean(cos_loss(model_out[i], target[i]))  # -1 * l.get('function')(model_out[i], target[i])
                elif l.get('type') == 'VGG':
                    loss = init_net(VGGLoss(), gpu_ids=args.CUDA_VISIBLE_DEVICES)(model_out[i], target[i])
                elif l.get('type') == 'SSIM':
                    loss = 1 - l.get('function')(model_out[i], target[i])
                else:
                    loss = l.get('function')(model_out[i], target[i])
                effective_loss = l.get('weight') * loss
                losses[i] = effective_loss
                l['value'] = losses[i].item()
            elif l.get('type') == 'Total':
                loss_sum = losses.sum()
                l['value'] = loss_sum.item()

        return loss_sum


def cos_loss(tensor1, tensor2):
    dot_mul = torch.sum(torch.mul(tensor1, tensor2), dim=1)
    tensor1_norm = torch.pow(torch.sum(torch.pow(tensor1, 2), dim=1) + 0.0001, 0.5)
    tensor2_norm = torch.pow(torch.sum(torch.pow(tensor2, 2), dim=1) + 0.0001, 0.5)
    loss = dot_mul / (tensor1_norm * tensor2_norm)
    return 1 - torch.mean(loss)


CONTENT_LAYER = 'relu_16'
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGLoss(torch.nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        device = torch.device('cpu' if args.b_cpu else 'cuda')
        self.VGG_19 = vgg_19().to(device)
        self.L1_loss = torch.nn.L1Loss()

    def forward(self, img1, img2):
        img1_vgg = self.VGG_19(normalize_batch(img1))
        img2_vgg = self.VGG_19(normalize_batch(img2))
        loss_vgg = self.L1_loss(img1_vgg, img2_vgg)
        return loss_vgg


class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.features = make_layers(cfgs['E'], batch_norm=False)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.load_state_dict(torch.load('./ckpt/vgg19.pth'))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def vgg_19():
    device = torch.device('cpu' if args.b_cpu else 'cuda')
    vgg_19 = VGG().features
    model = nn.Sequential()

    i = 0
    for layer in vgg_19.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        if name == CONTENT_LAYER:
            break

    for param in model.parameters():
        param.requires_grad = False

    for param in vgg_19.parameters():
        param.requires_grad = False

    return model

def normalize_batch(batch):
    batch = batch.cuda()
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


def init_net(net, init_type='default', init_gain=0.02, gpu_ids=[]):
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if init_type != 'default' and init_type is not None:
        init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 \
                or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'uniform':
                nn.init.uniform_(m.weight.data, b=init_gain)
            else:
                raise NotImplementedError('[%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
