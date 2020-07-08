from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from typing import Optional, List

from torch._jit_internal import Optional  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.intrinsic as nni
# from torch.nn.quantized.modules.utils import _quantize_weight
from torch.nn.modules.module import _addindent
from torch._ops import ops
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.quantized.modules.utils import _pair_from_first
from torch.nn.utils import fuse_conv_bn_weights

import torch.nn.intrinsic.qat as nniqat
from .noisy_layers import cal_range


def _quantize_weight(float_wt, observer):
    wt_scale, wt_zp = observer.calculate_qparams()
    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch.quantize_per_tensor(
            float_wt,
            float(wt_scale), int(wt_zp), observer.dtype)
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        wt_axis = observer.ch_axis
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.double), wt_zp.to(torch.int64), wt_axis, observer.dtype)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight


class Linear(torch.nn.Linear):
    r"""
    A quantized linear module with quantized tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`~torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias (Tensor): the non-learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized to zero.
        scale: `scale` parameter of output Quantized Tensor, type: double
        zero_point: `zero_point` parameter for output Quantized Tensor, type: long

    Examples::

        >>> m = nn.quantized.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, torch.quint8)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _version = 3
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, bias_=True, dtype=torch.qint8):
        super(Linear, self).__init__(in_features=in_features,
                                     out_features=out_features, bias=bias_)
        # We don't muck around with buffers or attributes or anything here
        # to keep the module simple. *everything* is simply a Python attribute.
        # Serialization logic is explicitly handled in the below serialization and
        # deserialization modules
        # self.in_features = in_features
        # self.out_features = out_features
        # self.bias = None
        # if bias_:
        #     self.bias = torch.zeros(out_features, dtype=torch.float)

        # if dtype == torch.qint8:
        #     self.weight = torch._empty_affine_quantized(
        #         [out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8)
        # elif dtype == torch.float16:
        #     self.weight = torch.zeros([out_features, in_features], dtype=torch.float)
        # else:
        #     raise RuntimeError('Unsupported dtype specified for quantized Linear!')

        self.scale = 1.0
        self.zero_point = 0
        self.quant_min = -128
        self.quant_max = 127

    def _get_name(self):
        return 'QuantizedLinear'

    def extra_repr(self):
        return 'in_features={}, out_features={}, scale={}, zero_point={}, qscheme={}'.format(
            self.in_features, self.out_features, self.scale, self.zero_point, self.weight().qscheme()
        )

    def __repr__(self):
        # We don't want to show `LinearPackedParams` children, hence custom
        # `__repr__`. This is the same as nn.Module.__repr__, except the check
        # for the `LinearPackedParams`.
        # You should still override `extra_repr` to add more info.
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def forward(self, x):
        # return torch.ops.quantized.linear(
        #     x, self._packed_params._packed_params, self.scale, self.zero_point)
        return torch.fake_quantize_per_tensor_affine(
            F.linear(x, self.weight, self.bias),
            scale=self.scale, zero_point=self.zero_point,
            quant_min=self.quant_min, quant_max=self.quant_max
        )

    # ===== Serialization methods =====
    # The special consideration here is that we have to unpack the weights into their
    # regular QTensor form for serialization. Packed weights should not live
    # outside the process in which they were created, rather they should be derived
    # from the QTensor weight.
    #
    # Version 1
    #   self
    #   |--- scale : float
    #   |--- zero_point : int
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # Version 2
    #   self
    #   |--- scale : float
    #   |--- zero_point : int
    #   |--- _packed_params : Module
    #        |--- weight : Tensor
    #        |--- bias : Tensor
    #
    # Version 3
    #   self
    #   |--- scale : float
    #   |--- zero_point : int
    #   |--- _packed_params : Module
    #        |--- _packed_params : (Tensor, Tensor) representing weight, bias
    #                              of LinearPackedParams C++ struct
    #
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # pylint: disable=not-callable
        super(Linear, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized QTensor
    # weight into its packed format for use by the FBGEMM ops.
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')

        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')

        version = local_metadata.get('version', None)
        # if version is None or version == 1:
        #     # We moved the parameters into a LinearPackedParameters submodule
        #     weight = state_dict.pop(prefix + 'weight')
        #     bias = state_dict.pop(prefix + 'bias')
        #     state_dict.update({prefix + '_packed_params.weight': weight,
        #                        prefix + '_packed_params.bias': bias})

        super(Linear, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                  missing_keys, unexpected_keys, error_msgs)

    def set_weight_bias(self, w, b):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        # quantize weight and bias matching their scale and zero_point
        self.weight.data.copy_(w)
        self.bias.data.copy_(b)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a float module -or qparams_dict-

        Args:
            mod (Module): a float module, either produced by torch.quantization
                          utilities or provided by the user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' nnq.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(
            mod, 'qconfig'), 'Input float module must have qconfig defined'
        if type(mod) == nni.LinearReLU:
            activation_post_process = mod[1].activation_post_process
            mod = mod[0]
        else:
            activation_post_process = mod.activation_post_process
        weight_post_process = mod.qconfig.weight()

        qbias = mod.bias
        weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()
        # assert dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'
        # TODO: explicitly change to quantize per tensor
        qweight = _quantize_weight(
            mod.weight.float(), weight_post_process).dequantize()
        if mod.bias is not None:
            bias_post_process = mod.qconfig.weight()
            bias_post_process(mod.bias)
            qbias = _quantize_weight(
                mod.bias.float(), bias_post_process).dequantize()

        qlinear = cls(mod.in_features, mod.out_features, dtype=dtype)
        qlinear.set_weight_bias(qweight, qbias)
        qlinear.scale = float(act_scale)
        qlinear.zero_point = int(act_zp)
        qlinear.quant_min = activation_post_process.qmin
        qlinear.quant_max = activation_post_process.qmax
        return qlinear


class _ConvNd(nn.modules.conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode='zeros'):
        super(_ConvNd, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, transposed=transposed,
            output_padding=output_padding, groups=groups, bias=bias,
            padding_mode=padding_mode
        )
        # if padding_mode != 'zeros':
        #     raise NotImplementedError(
        #         "Currently only zero-padding is supported by quantized conv")
        # if in_channels % groups != 0:
        #     raise ValueError('in_channels must be divisible by groups')
        # if out_channels % groups != 0:
        #     raise ValueError('out_channels must be divisible by groups')
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding
        # self.dilation = dilation
        # self.transposed = transposed
        # self.output_padding = output_padding
        # self.groups = groups
        # self.padding_mode = padding_mode
        # Initialize as NCHW. set_weight will internally transpose to NHWC.
        # qweight = torch._empty_affine_quantized(
        #     [out_channels, in_channels // self.groups] + list(kernel_size),
        #     scale=1, zero_point=0, dtype=torch.qint8)
        # bias_float = (
        #     torch.zeros(out_channels, dtype=torch.float) if bias else None)

        # self.set_weight_bias(qweight, bias_float)
        self.scale = 1.0
        self.zero_point = 0
        self.quant_min = -128
        self.quant_max = 127

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, scale={scale}, zero_point={zero_point}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias() is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    # ===== Serialization methods =====
    # The special consideration here is that we have to unpack the weights into
    # their regular QTensor form for serialization. Packed weights should not
    # live outside the process in which they were created, rather they should be
    # derived from the QTensor weight.
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # TODO: maybe change to this when https://github.com/pytorch/pytorch/pull/32958 is landed
    #   self
    #   |--- _packed_params : Conv2dPackedParamsBase or Conv3dPackedParamsBase
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # pylint: disable=not-callable
        super(_ConvNd, self)._save_to_state_dict(
            destination, prefix, keep_vars)
        (w, b) = self._weight_bias()
        destination[prefix + 'weight'] = w
        destination[prefix + 'bias'] = b
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

    # @torch.jit.export
    # def __getstate__(self):
    #     if not torch.jit.is_scripting():
    #         raise RuntimeError(
    #             'torch.save() is not currently supported for quantized modules.'
    #             ' See https://github.com/pytorch/pytorch/issues/24045.'
    #             ' Please use state_dict or torch.jit serialization.')
    #     (w, b) = self._weight_bias()
    #     return (
    #         self.in_channels,
    #         self.out_channels,
    #         self.kernel_size,
    #         self.stride,
    #         self.padding,
    #         self.dilation,
    #         self.transposed,
    #         self.output_padding,
    #         self.groups,
    #         self.padding_mode,
    #         self.weight,
    #         self.bias,
    #         self.scale,
    #         self.zero_point,
    #         self.training
    #     )

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized
    # QTensor weight into its packed format for use by the FBGEMM ops.
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.set_weight_bias(
            state_dict[prefix + 'weight'], state_dict[prefix + 'bias'])
        state_dict.pop(prefix + 'weight')
        state_dict.pop(prefix + 'bias')
        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')
        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')
        super(_ConvNd, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys,
            unexpected_keys, error_msgs)

    def set_weight_bias(self, w, b):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        self.weight.data.copy_(w)
        self.bias.data.copy_(b)
    # @torch.jit.export
    # def __setstate__(self, state):
    #     self.in_channels = state[0]
    #     self.out_channels = state[1]
    #     self.kernel_size = state[2]
    #     self.stride = state[3]
    #     self.padding = state[4]
    #     self.dilation = state[5]
    #     self.transposed = state[6]
    #     self.output_padding = state[7]
    #     self.groups = state[8]
    #     self.padding_mode = state[9]
    #     self.set_weight_bias(state[10], state[11])
    #     self.scale = state[12]
    #     self.zero_point = state[13]
    #     self.training = state[14]


class _OrdinaryConvNd(_ConvNd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_float(cls, mod):
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.quantization
              utilities or provided by the user
        """
        assert type(mod) == cls._FLOAT_MODULE, \
            ' nnq.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), \
            'Input float module must have qconfig defined.'

        qbias = mod.bias

        # activation_post_process = mod.activation_post_process
        weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        act_scale, act_zp = mod.activation_post_process.calculate_qparams()
        # assert weight_post_process.dtype == torch.qint8, 'Weight observer must have a dtype of qint8'
        qweight = _quantize_weight(
            mod.weight.float(), weight_post_process).dequantize()
        if mod.bias is not None:
            bias_post_process = mod.qconfig.weight()
            bias_post_process(mod.bias)
            qbias = _quantize_weight(
                mod.bias.float(), bias_post_process).dequantize()
        qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                    mod.stride, mod.padding, mod.dilation, mod.groups,
                    mod.bias is not None, mod.padding_mode)
        qconv.set_weight_bias(qweight, qbias)
        qconv.scale = float(act_scale)
        qconv.zero_point = int(act_zp)
        qconv.quant_min = mod.activation_post_process.qmin
        qconv.quant_max = mod.activation_post_process.qmax

        return qconv


class Conv1d(_OrdinaryConvNd, _ConvNd):
    r"""Applies a 1D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv1d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv1d` for other attributes.

    Examples::

        >>> m = nn.quantized.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> # quantize input to quint8
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0,
                                                dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE = nn.Conv1d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)

        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

    def _get_name(self):
        return 'QuantizedConv1d'

    # def _weight_bias(self):
    #     w, b = torch.ops.quantized.conv1d_unpack(self._packed_params)
    #     return w, b

    # def weight(self):
    #     return self._weight_bias()[0]

    # def bias(self):
    #     return self._weight_bias()[1]

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        # return ops.quantized.conv1d(input, self._packed_params, self.scale, self.zero_point)
        if self.padding_mode != 'zeros':
            out = F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                           self.weight, self.bias, self.stride,
                           _single(0), self.dilation, self.groups)
        else:
            out = F.conv1d(input, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)
        return torch.fake_quantize_per_tensor_affine(
            out,
            scale=self.scale, zero_point=self.zero_point,
            quant_min=self.quant_min, quant_max=self.quant_max
        )


class Conv2d(_OrdinaryConvNd, _ConvNd):
    r"""Applies a 2D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv2d` for other attributes.

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> # quantize input to quint8
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """
    _FLOAT_MODULE = nn.Conv2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def _get_name(self):
        return 'QuantizedConv2d'

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != 'zeros':
            out = F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                           self.weight, self.bias, self.stride,
                           _pair(0), self.dilation, self.groups)
        else:
            out = F.conv2d(input, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)

        return torch.fake_quantize_per_tensor_affine(
            out,
            scale=self.scale, zero_point=self.zero_point,
            quant_min=self.quant_min, quant_max=self.quant_max
        )


class Conv3d(_OrdinaryConvNd, _ConvNd):
    r"""Applies a 3D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv3d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv3d` for other attributes.

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), dilation=(1, 2, 2))
        >>> input = torch.randn(20, 16, 56, 56, 56)
        >>> # quantize input to quint8
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """
    _FLOAT_MODULE = nn.Conv3d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode)

    def _get_name(self):
        return 'QuantizedConv3d'

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        if self.padding_mode != 'zeros':
            out = F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                           self.weight, self.bias, self.stride, _triple(0),
                           self.dilation, self.groups)
        else:
            out = F.conv3d(input, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)
        return torch.fake_quantize_per_tensor_affine(
            out,
            scale=self.scale, zero_point=self.zero_point,
            quant_min=self.quant_min, quant_max=self.quant_max
        )


class ReLU(torch.nn.ReLU):
    r"""Applies quantized rectified linear unit function element-wise:

    :math:`\text{ReLU}(x)= \max(x_0, x)`, where :math:`x_0` is the zero point.

    Please see https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU
    for more documentation on ReLU.

    Args:
        inplace: (Currently not supported) can optionally do the operation in-place.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.quantized.ReLU()
        >>> input = torch.randn(2)
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(inplace)
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)

    def _get_name(self):
        return 'QuantizedReLU'
