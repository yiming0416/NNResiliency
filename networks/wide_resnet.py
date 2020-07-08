import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .noisy_layers import *
from .quantization import CustomFakeQuantize, get_activation_quant, enable_fake_quant, enable_observer, disable_fake_quant, disable_observer
from .network_utils import num_parameters, children_of_class
# TODO: move this along with the set_mu_list and set_sigma_list to NoisyModule base class
from itertools import cycle
from typing import Iterable

import sys
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, sigma=False):
    """3x3 convolution with padding"""
    return NoisyConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True, sigma=sigma)


def wide_conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('NoisyBN') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, sigma=[None, None], quantization_levels=1024):
        super(wide_basic, self).__init__()
        self.bn1 = NoisyBN(in_planes)
        self.conv1 = NoisyConv2d(
            in_planes, planes, kernel_size=3, padding=1, bias=True, sigma=sigma[0])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = NoisyBN(planes)
        self.conv2 = NoisyConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, sigma=sigma[1])
        self.out_quant = get_activation_quant(
            quantization_levels, enable=False)

        if stride != 1 or in_planes != planes:
            self.downsample = NoisyConv2d(
                in_planes, planes, kernel_size=1, stride=stride, bias=True)
        else:
            self.downsample = NoisyIdentity()

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + self.downsample(x)
        out = self.out_quant(out)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, in_channels=3, sigma_list=None, quantization_levels=1024):
        super(WideResNet, self).__init__()

        self.name = "WideResNet-" + str(depth) + "-" + str(widen_factor)
        self.in_planes = 16

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        if sigma_list is not None:
            try:
                sigma_list = float(sigma_list)
                sigma_list = np.repeat(sigma_list, 1 + n*6)
            except TypeError:
                if len(sigma_list) != 1 + n*6:
                    raise ValueError(
                        'The length of `sigma_list` is not compatible')
                else:
                    sigma_list = np.array(sigma_list)
        else:
            sigma_list = np.repeat(sigma_list, 1 + n*6)

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(in_channels, nStages[0], sigma=sigma_list[0])
        self.conv1_quant = get_activation_quant(
            quantization_levels, enable=False)
        self.layer1 = self._wide_layer(
            wide_basic, nStages[1], n, dropout_rate, stride=1, sigma_list=sigma_list[1:1+n*2])
        self.layer2 = self._wide_layer(
            wide_basic, nStages[2], n, dropout_rate, stride=2, sigma_list=sigma_list[1+n*2:1+n*4])
        self.layer3 = self._wide_layer(
            wide_basic, nStages[3], n, dropout_rate, stride=2, sigma_list=sigma_list[1+n*4:1+n*6])
        self.bn1 = NoisyBN(nStages[3])
        self.before_linear_quant = get_activation_quant(
            quantization_levels, enable=False)
        self.linear = NoisyLinear(nStages[3], num_classes)

        s = '[%s] Num parameters: %d' % (self.name, num_parameters(self))
        print(s)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, sigma_list=None):
        if sigma_list is None:
            sigma_list = np.repeat(None, num_blocks * 2)
        assert (len(sigma_list) == num_blocks * 2)

        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes,
                                dropout_rate, stride, sigma=sigma_list[2*i:2*i+2]))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_quant(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = self.before_linear_quant(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def enable_quantization(self, flag: bool = True) -> None:
        for quant in children_of_class(self, CustomFakeQuantize):
            if flag:
                quant.enable_fake_quant()
                # enable_fake_quant(quant)
                # enable_observer(quant)
            else:
                quant.disable_fake_quant()
                # disable_fake_quant(quant)
                # disable_observer(quant)

    def set_quantization_level(self, quantization_levels: int) -> None:
        for quant in children_of_class(self, CustomFakeQuantize):
            quant.set_qmin_qmax(0, quantization_levels - 1)

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

    def set_sigma_list(self, sigma_list: Iterable) -> None:
        noisy_layer_list = list(children_of_class(self, NoisyLayer))

        if sigma_list is None:
            sigma_list = [0]
        else:
            try:
                sigma_list = [float(sigma_list)]
            except:
                pass
        for l, s in zip(noisy_layer_list, cycle(sigma_list)):
            l.sigma = s

    # def set_sigma_list(self, sigma_list):
    #     """ Allow None, scalar, 1-length list or
    #     list with the length of the number of noisy layers
    #     """
    #     noisy_conv_list = list(children_of_class(self, NoisyConv2d))

    #     if sigma_list is None:
    #         conv_sigma_list = [sigma_list] * len(noisy_conv_list)
    #     else:
    #         try:
    #             conv_sigma_list = float(sigma_list)
    #             conv_sigma_list = [conv_sigma_list] * len(noisy_conv_list)
    #         except:
    #             assert type(sigma_list) == list
    #             if len(sigma_list) == 1:
    #                 conv_sigma_list = sigma_list * len(noisy_conv_list)
    #             else:
    #                 assert len(conv_sigma_list) == len(noisy_conv_list)

    #     for m, s in zip(noisy_conv_list, conv_sigma_list):
    #         m.sigma = s

    #     noisy_fc_list = list(children_of_class(self, NoisyLinear))

    #     if sigma_list is None:
    #         fc_sigma_list = [sigma_list] * len(noisy_fc_list)
    #     else:
    #         try:
    #             fc_sigma_list = float(sigma_list)
    #             fc_sigma_list = [fc_sigma_list] * len(noisy_fc_list)
    #         except:
    #             assert type(sigma_list) == list
    #             if len(sigma_list) == 1:
    #                 fc_sigma_list = sigma_list * len(noisy_fc_list)
    #             else:
    #                 assert len(fc_sigma_list) == len(noisy_fc_list)

    #     for m, s in zip(noisy_fc_list, fc_sigma_list):
    #         m.sigma = s

    #     noisy_id_list = list(children_of_class(self, NoisyIdentity))

    #     if sigma_list is None:
    #         id_sigma_list = [sigma_list] * len(noisy_id_list)
    #     else:
    #         try:
    #             id_sigma_list = float(sigma_list)
    #             id_sigma_list = [id_sigma_list] * len(noisy_id_list)
    #         except:
    #             assert type(sigma_list) == list
    #             if len(sigma_list) == 1:
    #                 id_sigma_list = sigma_list * len(noisy_id_list)
    #             else:
    #                 assert len(id_sigma_list) == len(noisy_id_list)

    #     for m, s in zip(noisy_id_list, id_sigma_list):
    #         m.sigma = s

    #     noisy_bn_list = list(children_of_class(self, NoisyBN))

    #     if sigma_list is None:
    #         bn_sigma_list = [sigma_list] * len(noisy_bn_list)
    #     else:
    #         try:
    #             bn_sigma_list = float(sigma_list)
    #             bn_sigma_list = [bn_sigma_list] * len(noisy_bn_list)
    #         except:
    #             assert type(sigma_list) == list
    #             if len(sigma_list) == 1:
    #                 bn_sigma_list = sigma_list * len(noisy_bn_list)
    #             else:
    #                 assert len(bn_sigma_list) == len(noisy_bn_list)

    #     for m, s in zip(noisy_bn_list, bn_sigma_list):
    #         m.sigma = s
