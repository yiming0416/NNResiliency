import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .noisy_layers import *
from .quantization import CustomFakeQuantize, get_activation_quant, enable_fake_quant, enable_observer, disable_fake_quant, disable_observer
from .network_utils import children_of_class, num_parameters
from typing import Iterable
from itertools import cycle


def conv3x3(in_planes, out_planes, stride=1, sigma=False):
    return NoisyConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, sigma=sigma)

def conv1x1(in_planes, out_planes, stride=1, sigma=False):
    return NoisyConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=1, bias=False, sigma=sigma)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        #nn.init.constant_(m.bias, 0)

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_dropout=True,
                 dropout_rate=0.3, quantization_levels=256):
        super(BasicBlock, self).__init__()

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = NoisyBN(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = NoisyBN(planes)
        self.out_quant = get_activation_quant(quantization_levels, enable=False)
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                NoisyConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, padding=0, bias=False),
                NoisyBN(self.expansion*planes)                         
            )
        else:
            self.downsample = NoisyIdentity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_dropout:
            out = self.dropout(out)
        out += self.downsample(x)
        out = F.relu(out)
        out = self.out_quant(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_dropout=True,
                 dropout_rate=0.3, quantization_levels=256):
        super(Bottleneck, self).__init__()

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = NoisyBN(planes)
        self.conv2 = conv1x1(planes, planes, stride=stride)
        self.bn2 = NoisyBN(planes)
        self.conv3 = conv1x1(planes, self.expansion*planes)
        self.bn3 = NoisyBN(self.expansion*planes)
        self.out_quant = get_activation_quant(quantization_levels, enable=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                NoisyConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, padding=0, bias=False),
                NoisyBN(self.expansion*planes)                         
            )        
        else:
            self.downsample = NoisyIdentity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.use_dropout:
            out = self.dropout(out)
        out += self.downsample(x)
        out = F.relu(out)
        out = self.out_quant(out)

        return out

class ResNet(nn.Module):
    def __init__(self, depth, num_classes, sigma_list=None, use_dropout=True, dropout_rate=0.3, in_channel=3):
        super(ResNet, self).__init__()
        self.name = "ResNet-" + str(depth)
        self.in_planes = 64
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        block, num_blocks = cfg(depth)

        self.conv1 = NoisyConv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)        
        self.bn1 = NoisyBN(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, use_dropout, dropout_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, use_dropout, dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, use_dropout, dropout_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, use_dropout, dropout_rate)        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = NoisyLinear(512*block.expansion, num_classes)
        self.set_sigma_list(sigma_list)

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self))
        print(s)

    def _make_layer(self, block, planes, num_blocks, stride, use_dropout, dropout_rate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_dropout, dropout_rate))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def set_mu_list(self, mu_list: Iterable) -> None:
        noisy_layer_list = list(children_of_class(self, NoisyLayer))

        if mu_list is None:
            mu_list = [0]
        else:
            try:
                mu_list = [float(mu_list)]
            except:
                pass

        for l, m in zip(noisy_layer_list, cycle(mu_list)):
            l.mu = m
    
    def enable_quantization(self, flag: bool=True) -> None:
        for quant in children_of_class(self, CustomFakeQuantize):
            if flag:
                quant.enable_fake_quant()
                # quant.enable_observer()
            else:
                quant.disable_fake_quant()
                # quant.disable_observer()

    def set_quantization_level(self, quantization_levels: int) -> None:
        for quant in children_of_class(self, CustomFakeQuantize):
            quant.set_qmin_qmax(0, quantization_levels - 1)

    def set_sigma_list(self, sigma_list):
        """ Allow None, scalar, 1-length list or
        list with the length of the number of noisy layers
        """
        noisy_conv_list = list(children_of_class(self, NoisyConv2d))

        if sigma_list is None:
            conv_sigma_list = [0] * len(noisy_conv_list)
        else:
            try:
                conv_sigma_list = float(sigma_list)
                conv_sigma_list = [conv_sigma_list] * len(noisy_conv_list)
            except:
                assert type(sigma_list) == list
                if len(sigma_list) == 1:
                    conv_sigma_list = sigma_list * len(noisy_conv_list)
                else:
                    assert len(conv_sigma_list) == len(noisy_conv_list)

        for m, s in zip(noisy_conv_list, conv_sigma_list):
            m.sigma = s

        noisy_fc_list = list(children_of_class(self, NoisyLinear))

        if sigma_list is None:
            fc_sigma_list = [0] * len(noisy_fc_list)
        else:
            try:
                fc_sigma_list = float(sigma_list)
                fc_sigma_list = [fc_sigma_list] * len(noisy_fc_list)
            except:
                assert type(sigma_list) == list
                if len(sigma_list) == 1:
                    fc_sigma_list = sigma_list * len(noisy_fc_list)
                else:
                    assert len(fc_sigma_list) == len(noisy_fc_list)

        for m, s in zip(noisy_fc_list, fc_sigma_list):
            m.sigma = s

        noisy_id_list = list(children_of_class(self, NoisyIdentity))
        
        if sigma_list is None:
            id_sigma_list = [0] * len(noisy_id_list)
        else:
            try:
                id_sigma_list = float(sigma_list)
                id_sigma_list = [id_sigma_list] * len(noisy_id_list)
            except:
                assert type(sigma_list) == list
                if len(sigma_list) == 1:
                    id_sigma_list = sigma_list * len(noisy_id_list)
                else:
                    assert len(id_sigma_list) == len(noisy_id_list)

        for m, s in zip(noisy_id_list, id_sigma_list):
            m.sigma = s

        noisy_bn_list = list(children_of_class(self, NoisyBN))

        if sigma_list is None:
            bn_sigma_list = [0] * len(noisy_bn_list)
        else:
            try:
                bn_sigma_list = float(sigma_list)
                bn_sigma_list = [bn_sigma_list] * len(noisy_bn_list)
            except:
                assert type(sigma_list) == list
                if len(sigma_list) == 1:
                    bn_sigma_list = sigma_list * len(noisy_bn_list)
                else:
                    assert len(bn_sigma_list) == len(noisy_bn_list)

        for m, s in zip(noisy_bn_list, bn_sigma_list):
            m.sigma = s
