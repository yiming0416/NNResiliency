# Copied from torch/quantization/observer.py
# Modified to allow more than 256 levels (and TODO future quantization schemes)
# See also https://medium.com/@karanbirchahal/aggressive-quantization-how-to-run-mnist-on-a-4-bit-neural-net-using-pytorch-5703f3faa599
import torch
from torch.quantization import *
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
import torch.nn.quantized as nnq
from . import quantized_layers as mynnq

import torch.nn.quantized.dynamic as nnqd
import torch.nn.qat as nnqat

from .observers import *
from .noisy_layers import *

class CustomFakeQuantize(FakeQuantize):
    def __init__(self, observer=CustomMovingAverageMinMaxObserver, quant_min=0, quant_max=255, **observer_kwargs):
        super(CustomFakeQuantize, self).__init__(observer, quant_min, quant_max, **observer_kwargs)
        self.activation_post_process.set_qmin_qmax(quant_min, quant_max) 
        self.observer_cls = observer.with_args(**observer_kwargs)

    def set_qmin_qmax(self, quant_min, quant_max):
        self.quant_min = quant_min
        self.quant_max = quant_max
        device = next(self.activation_post_process.buffers()).device # need to preserve the device where the observer is
        self.activation_post_process = self.observer_cls() # need to re-initialize the observer
        self.activation_post_process.set_qmin_qmax(quant_min, quant_max)
        self.activation_post_process.to(device)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if prefix + 'scale' in state_dict:
            self.scale = state_dict.pop(prefix + 'scale')
        if prefix + 'zero_point' in state_dict:
            self.zero_point = state_dict.pop(prefix + 'zero_point')
        Module._load_from_state_dict(
            self, state_dict, prefix, local_metadata, False,
            missing_keys, unexpected_keys, error_msgs)

# based on 'fbgemm' in qconfig
# weight_fake_quant_dummy =\
#     CustomFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
#                            quant_min=-128,
#                            quant_max=127,
#                            dtype=torch.qint8,
#                            qscheme=torch.per_channel_symmetric,
#                            reduce_range=False,
#                            ch_axis=0)

def get_activation_quant(nlevel: int, enable: bool=True):
    quant = CustomFakeQuantize.with_args(
        observer=CustomMovingAverageMinMaxObserver,
        dtype=torch.qint32,
        quant_min=0,
        quant_max=nlevel - 1,
        reduce_range=False)()
    if not enable:
        quant.disable_fake_quant()
        # disable_observer(quant) # Do we disable observer or not?
    return quant

def get_qconfig(nlevel_weight: int, nlevel_activation: int) -> QConfig:
    activation_observer = CustomHistogramObserver.with_args(
        quant_min=0, quant_max=nlevel_activation-1, reduce_range=False, dtype=torch.qint32
    )
    weight_observer = CustomMinMaxObserver.with_args(
        quant_min=0, quant_max=nlevel_weight-1,# dtype=torch.qint8,
        dtype=torch.qint32,
        # qscheme=torch.per_channel_symmetric
        qscheme=torch.per_tensor_affine
    )
    return QConfig(activation=activation_observer, weight=weight_observer)

CUSTOM_MODULE_MAPPING = {
    nn.Linear: mynnq.Linear,
    nn.ReLU: mynnq.ReLU,
    nn.ReLU6: nnq.ReLU6,
    nn.Hardswish: nnq.Hardswish,
    nn.Conv1d: mynnq.Conv1d,
    nn.Conv2d: mynnq.Conv2d,
    nn.Conv3d: mynnq.Conv3d,
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nn.LayerNorm: nnq.LayerNorm,
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    # Wrapper Modules:
    nnq.FloatFunctional: nnq.QFunctional,
    # Intrinsic modules:
    nni.ConvReLU1d: nniq.ConvReLU1d,
    nni.ConvReLU2d: nniq.ConvReLU2d,
    nni.ConvReLU3d: nniq.ConvReLU3d,
    nni.LinearReLU: nniq.LinearReLU,
    nni.BNReLU2d: nniq.BNReLU2d,
    nni.BNReLU3d: nniq.BNReLU3d,
    nniqat.ConvReLU2d: nniq.ConvReLU2d,
    nniqat.LinearReLU: nniq.LinearReLU,
    nniqat.ConvBn2d: nnq.Conv2d,
    nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
    # QAT modules:
    nnqat.Linear: nnq.Linear,
    nnqat.Conv2d: nnq.Conv2d,
    nnqat.Hardswish: nnq.Hardswish,
    # Noisy modules:
    NoisyLinear: mynnq.Linear,
    NoisyConv2d: mynnq.Conv2d,
    # NoisyIdentity: nn.Identity,
    # Custom:
    # CustomFakeQuantize: nn.Identity,
}

CUSTOM_QCONFIG_PROPAGATE_WHITE_LIST = (DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST | set(CUSTOM_MODULE_MAPPING.keys()))