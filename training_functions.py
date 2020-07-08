import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from networks import set_gaussian_noise, set_uniform_noise, set_fixtest, disable_observer, disable_fake_quant, get_qconfig, CUSTOM_MODULE_MAPPING, CUSTOM_QCONFIG_PROPAGATE_WHITE_LIST, children_of_class, NoisyLayer, CustomFakeQuantize
import pandas as pd
import numpy as np
import argparse
from typing import Union, Iterable


def prepare_network_perturbation(
        net, noise_type: str = 'gaussian', fixtest: bool = False,
        perturbation_level: Union[None, float, Iterable[float]] = None,
        perturbation_mean: Union[None, float, Iterable[float]] = None):
    """Set the perturbation and quantization of the network in-place
    """
    if noise_type == 'gaussian':
        net.apply(set_gaussian_noise)
        if isinstance(net, nn.DataParallel):
            net.module.set_sigma_list(perturbation_level)
            net.module.set_mu_list(perturbation_mean)
        else:
            net.set_sigma_list(perturbation_level)
            net.set_mu_list(perturbation_mean)
    elif noise_type == 'uniform':
        net.apply(set_uniform_noise)
        if isinstance(net, nn.DataParallel):
            net.module.set_sigma_list(1)
        else:
            net.set_sigma_list(1)

    if fixtest:
        net.apply(set_fixtest)


def prepare_network_quantization(
        net, num_quantization_levels: int, calibration_dataloader: torch.utils.data.DataLoader,
        qat: bool = False, num_calibration_batchs: int = 10):  # The last two arguments are redundant for now
    if num_quantization_levels is None:
        return
    # Specify quantization configuration
    net.set_quantization_level(num_quantization_levels)
    net.enable_quantization(False)
    # Calibrate with the test set
    net.eval()
    device = next(net.parameters()).device
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(calibration_dataloader):
            inputs, targets = inputs.to(
                device=device), targets.to(device=device)
            outputs = net(inputs)
    print('Post Training Quantization: Calibration done')
    net.enable_quantization()
    for quant in children_of_class(net, CustomFakeQuantize):
        quant.disable_observer()


def quantize_network(net: nn.Module, num_quantization_levels: int, calibration_dataloader: torch.utils.data.DataLoader):
    net.qconfig = get_qconfig(num_quantization_levels, 2*num_quantization_levels)

    for noisy_layer in children_of_class(net, NoisyLayer):
        noisy_layer.to_original()
    for activation_quant in children_of_class(net, CustomFakeQuantize):
        disable_observer(activation_quant)
        disable_fake_quant(activation_quant)
    torch.quantization.prepare(
        net, inplace=True,
        white_list=CUSTOM_QCONFIG_PROPAGATE_WHITE_LIST
    )
    # Calibrate with the given set
    net.eval()
    device = next(net.parameters()).device
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(calibration_dataloader):
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            outputs = net(inputs)
    torch.quantization.convert(
        net, inplace=True,
        # modify below to choose whether to use custom quantized layers
        mapping=CUSTOM_MODULE_MAPPING
    )

    # print('Quantization Config:', net.qconfig)
    # if qat:
    #     torch.quantization.prepare_qat(net, inplace=True)
    #     test(net, -1, calibration_dataloader)
    # else:
    #     torch.quantization.prepare(net, inplace=True)
    #     # Calibrate first
    #     print('Post Training Quantization Prepare: Inserting Observers')

    #     # Calibrate with the test set TODO: use the training set to calibrate
    #     test(net, -1, calibration_dataloader)
    #     print('Post Training Quantization: Calibration done')

    #     # Convert to quantized model
    #     torch.quantization.convert(net, inplace=True)
    #     print('Post Training Quantization: Convert done')


class Clipper(dict):  # inherit dict to be serializable
    def __init__(self, max_norm: float = None, decay_factor: float = 1, decay_interval: int = None, max_decay_times: int = None):
        assert max_norm is None or max_norm > 0
        assert decay_factor > 0 and decay_factor <= 1
        assert decay_interval is None or decay_interval > 0
        assert max_decay_times is None or max_decay_times > 0
        self.clip_function = clip_grad_norm_
        self.max_norm = max_norm
        self.steps = 0
        self.decay_interval = np.inf if decay_interval is None else decay_interval
        self.decay_factor = decay_factor
        self.max_decay_times = np.inf if max_decay_times is None else max_decay_times
        self.decay_counter = 0
        super(Clipper, self).__init__(
            self,
            clip_function=self.clip_function.__name__,
            max_norm=self.max_norm,
            decay_interval=self.decay_interval,
            decay_factor=self.decay_factor,
            max_decay_times=self.max_decay_times,
        )

    def step(self):
        self.steps += 1
        if self.steps % self.decay_interval == 0 and self.decay_counter < self.max_decay_times:
            self.decay_counter += 1
            self.max_norm *= self.decay_factor

    def clip(self, parameters):
        if self.max_norm is not None:
            self.clip_function(parameters, self.max_norm)

    def __str__(self):
        keys = [
            "clip_function", "max_norm", "decay_interval", "decay_factor", "max_decay_times"
        ]
        string = ", ".join(["{}={}".format(k, getattr(self, k)) for k in keys])
        return "Clipper(" + string + ")"


def grad_clipper(string: str) -> Clipper:
    """The parser for the grad_clip"""
    clipper = Clipper()
    if string is None:
        return clipper

    toks = string.split(":")
    try:
        if len(toks) == 1:
            clipper = Clipper(float(toks[0]))
        elif len(toks) == 3:
            clipper = Clipper(float(toks[0]), float(toks[1]), int(toks[2]))
        elif len(toks) == 4:
            clipper = Clipper(float(toks[0]), float(
                toks[1]), int(toks[2]), int(toks[3]))
        else:
            msg = "Required format: <init_max_norm>[:<decay_factor>:<decay_interval>[:<max_decay_count>]]"
            raise argparse.ArgumentTypeError(msg)
    except Exception as e:
        msg = "Required format: <init_max_norm>[:<decay_factor>:<decay_interval>[:<max_decay_count>]]"
        raise argparse.ArgumentTypeError(msg)
    return clipper
