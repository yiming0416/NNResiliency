# Copied from torch/quantization/observer.py
# Modified to allow more than 256 levels (and TODO future quantization schemes)
# See also https://medium.com/@karanbirchahal/aggressive-quantization-how-to-run-mnist-on-a-4-bit-neural-net-using-pytorch-5703f3faa599
import torch
from torch.quantization import *

#: Modified from _ObserverBase
class CustomObserverBase(ObserverBase):
    r"""Internal common base for all qint/quint8 observers.

    This base is for commonly used paramters used internally.
    Users should use `~torch.quantization.observer.ObserverBase` as a base class
    for custom observers.

    Args:
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        reduce_range: Reduces the range of the quantized data type by 1 bit.
                      This is sometimes required to avoid instruction overflow.

    .. warning::

        :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.

    .. warning::

        :attr:`qscheme` can only take one of the following options:

        - ``torch.per_tensor_affine``
        - ``torch.per_tensor_symmetric``
        - ``torch.per_channel_affine``
        - ``torch.per_channel_symmetric``
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_max=None, quant_min=None):
        super(CustomObserverBase, self).__init__(dtype=dtype)
        self.qscheme = qscheme
        self.reduce_range = reduce_range

        self.eps = torch.finfo(torch.float32).eps
        assert self.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
            torch.per_channel_affine,
            torch.per_channel_symmetric,
        ), "Default Observer only works for per_tensor_affine, \
                per_tensor_symmetric, per_channel_affine and \
                per_channel_symmetric quantization scheme"
        assert self.dtype in (
            torch.qint8,
            torch.quint8,
            torch.qint32,
        ), "Custom Observer only works for qint8, quint8 and qint32 data type"

        self.set_qmin_qmax(quant_min, quant_max)

    def set_qmin_qmax(self, quant_min, quant_max):
        assert torch.iinfo(self.dtype).min <= quant_min, 'quant_min out of bound'
        assert quant_max <= torch.iinfo(self.dtype).max, 'quant_max out of bound'
        if quant_min is None or quant_max is None:
            if self.dtype == torch.qint8:
                if self.reduce_range:
                    self.qmin, self.qmax = -64, 63
                else:
                    self.qmin, self.qmax = -128, 127
            elif self.dtype == torch.quint8:
                if self.reduce_range:
                    self.qmin, self.qmax = 0, 127
                else:
                    self.qmin, self.qmax = 0, 255
            elif self.dtype == torch.qint32:
                if self.reduce_range:
                    self.qmin = torch.iinfo(torch.qint32).min // 2
                    self.qmax = torch.iinfo(torch.qint32).max // 2
                else:
                    self.qmin = torch.iinfo(torch.qint32).min
                    self.qmax = torch.iinfo(torch.qint32).max
        else:
            self.qmin, self.qmax = quant_min, quant_max

    def _calculate_per_channel_qparams(self, min_vals, max_vals):
        # type: (Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor, int]
        r"""Calculates the per channel quantization parameters, given min and max
        value tensors.

        Args:
            min_vals: Minimum values per channel
            max_vals: Maximum values per channel

        Returns:
            scales: Per channel scales tensor of shape (#channels,)
            zero_points: Per channel zero points tensor of shape (#channels,)
        """
        if min_vals is None or max_vals is None:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]), torch.tensor([0]), self.ch_axis

        for i in range(len(min_vals)):
            assert (
                min_vals[i] <= max_vals[i]
            ), "min {} should be less than max {}".format(min_vals[i], max_vals[i])

        scales = torch.empty(min_vals.size(), dtype=torch.float32)
        zero_points = torch.empty(min_vals.size(), dtype=torch.int64)

        for i in range(len(scales)):
            qparam = self._calculate_qparams(
                min_vals[i], max_vals[i]
            )
            scales[i] = float(qparam[0])
            zero_points[i] = int(qparam[1])

        return scales, zero_points, self.ch_axis

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        # type: (Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        r"""Calculates the per tensor quantization parameters, given the min/max.

        Args:
            min_val: Per tensor minimum value
            max_val: Per tensor maximum value

        Returns:
            scale: Scale as a tensor of shape (1,)
            zero_point: Zero point as a tensor of shape (1,)
        """

        if max_val is None or min_val is None:
            warnings.warn("Must run observer before calling calculate_qparams.\
                           Returning default scale and zero point.")
            return torch.tensor([1.0]), torch.tensor([0])

        assert min_val <= max_val, "min {} should be less than max {}".format(
            min_val, max_val
        )

        max_val, min_val = float(max_val), float(min_val)
        min_val = min(0.0, min_val)
        max_val = max(0.0, max_val)
        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
                max_val = max(-min_val, max_val)
                scale = max_val / ((self.qmax - self.qmin) / 2)
                scale = max(scale, self.eps)
                zero_point = 0 if self.dtype == torch.qint8 else 128
            else:
                scale = (max_val - min_val) / float(self.qmax - self.qmin)
                scale = max(scale, self.eps)
                zero_point = self.qmin - round(min_val / scale)
                zero_point = max(self.qmin, zero_point)
                zero_point = min(self.qmax, zero_point)
                zero_point = int(zero_point)

        return torch.tensor([scale]), torch.tensor([zero_point])


class CustomMinMaxObserver(CustomObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running min and max values.

    This observer uses the tensor min/max statistics to compute the quantization
    parameters. The module records the running minimum and maximum of incoming
    tensors, and uses this statistic to compute the quantization parameters.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit

    Given running min/max as :math:`x_\text{min}` and :math:`x_\text{max}`,
    scale :math:`s` and zero point :math:`z` are computed as:

    The running minimum/maximum :math:`x_\text{min/max}` is computed as:

    .. math::

        \begin{array}{ll}
        x_\text{min} &= \begin{cases}
            \min(X) & \text{if~}x_\text{min} = \text{None} \\
            \min\left(x_\text{min}, \min(X)\right) & \text{otherwise}
        \end{cases}\\
        x_\text{max} &= \begin{cases}
            \max(X) & \text{if~}x_\text{max} = \text{None} \\
            \max\left(x_\text{max}, \max(X)\right) & \text{otherwise}
        \end{cases}\\
        \end{array}

    where :math:`X` is the observed tensor.

    The scale :math:`s` and zero point :math:`z` are then computed as:

    .. math::

        \begin{aligned}
            \text{if Symmetric:}&\\
            &s = 2 \max(|x_\text{min}|, x_\text{max}) /
                \left( Q_\text{max} - Q_\text{min} \right) \\
            &z = \begin{cases}
                0 & \text{if dtype is qint8} \\
                128 & \text{otherwise}
            \end{cases}\\
            \text{Otherwise:}&\\
                &s = \left( x_\text{max} - x_\text{min}  \right ) /
                    \left( Q_\text{max} - Q_\text{min} \right ) \\
                &z = Q_\text{min} - \text{round}(x_\text{min} / s)
        \end{aligned}

    where :math:`Q_\text{min}` and :math:`Q_\text{max}` are the minimum and
    maximum of the quantized data type.

    .. warning:: Only works with ``torch.per_tensor_symmetric`` quantization scheme

    .. warning:: :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    __annotations__ = {
        "min_val": Optional[torch.Tensor],
        "max_val": Optional[torch.Tensor],
    }

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False):
        # For x86 quantized kernels, we need to ensure that the vpmaddubsw
        # instruction does not overflow. We allow for a reduce_range argument to
        # observers that reduces the quantized range to (0,127) or (-64, 63).
        # For more details see aten/src/ATen/native/quantized/cpu/qconv.cpp
        # This is not an optimal choice for non x86 backends as it loses a bit
        # of precision for activations.

        super(CustomMinMaxObserver, self).__init__(dtype=dtype,
                                                   qscheme=qscheme,
                                                   reduce_range=reduce_range)
        self.min_val = None
        self.max_val = None
        if self.qscheme == torch.per_tensor_symmetric and \
           self.reduce_range and \
           self.dtype == torch.quint8:
            raise NotImplementedError("Cannot reduce range for symmetric \
                                       quantization for quint8")

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val
        if min_val is None or max_val is None:
            min_val = torch.min(x)
            max_val = torch.max(x)
        else:
            min_val = torch.min(torch.min(x), min_val)
            max_val = torch.max(torch.max(x), max_val)
        self.min_val = min_val
        self.max_val = max_val
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(CustomMinMaxObserver, self)._save_to_state_dict(
            destination, prefix, keep_vars)
        destination[prefix + 'min_val'] = self.min_val
        destination[prefix + 'max_val'] = self.max_val

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        self.min_val = state_dict.pop(prefix + 'min_val')
        self.max_val = state_dict.pop(prefix + 'max_val')
        super(CustomMinMaxObserver,
              self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                          missing_keys, unexpected_keys, error_msgs)


class CustomMovingAverageMinMaxObserver(CustomMinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    moving average of the min and max values.

    This observer computes the quantization parameters based on the moving
    averages of minimums and maximums of the incoming tensors. The module
    records the average minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit

    The moving average min/max is computed as follows

    .. math::

        \begin{array}{ll}
                x_\text{min} = \begin{cases}
                    \min(X) & \text{if~}x_\text{min} = \text{None} \\
                    (1 - c) x_\text{min} + c \min(X) & \text{otherwise}
                \end{cases}\\
                x_\text{max} = \begin{cases}
                    \max(X) & \text{if~}x_\text{max} = \text{None} \\
                    (1 - c) x_\text{max} + c \max(X) & \text{otherwise}
                \end{cases}\\
        \end{array}

    where :math:`x_\text{min/max}` is the running average min/max, :math:`X` is
    is the incoming tensor, and :math:`c` is the ``averaging_constant``.

    The scale and zero point are then computed as in
    :class:`~torch.quantization.observer.MinMaxObserver`.

    .. note:: Only works with ``torch.per_tensor_affine`` quantization shceme.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    def __init__(self, averaging_constant=0.01, dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine, reduce_range=False):
        self.averaging_constant = averaging_constant
        super(CustomMovingAverageMinMaxObserver, self).__init__(dtype=dtype,
                                                                qscheme=qscheme,
                                                                reduce_range=reduce_range)

    def forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val
        if min_val is None or max_val is None:
            min_val = torch.min(x)
            max_val = torch.max(x)
        else:
            min_val = min_val + self.averaging_constant * \
                (torch.min(x) - min_val)
            max_val = max_val + self.averaging_constant * \
                (torch.max(x) - max_val)
        self.min_val = min_val
        self.max_val = max_val
        return x_orig


class CustomPerChannelMinMaxObserver(CustomObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        ch_axis: Channel axis
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit

    The quantization parameters are computed the same way as in
    :class:`~torch.quantization.observer.MinMaxObserver`, with the difference
    that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """
    __annotations__ = {
        "min_vals": Optional[torch.Tensor],
        "max_vals": Optional[torch.Tensor],
    }

    def __init__(self, ch_axis=0, dtype=torch.quint8,
                 qscheme=torch.per_channel_affine, reduce_range=False):
        super(CustomPerChannelMinMaxObserver, self).__init__(dtype=dtype,
                                                             qscheme=qscheme,
                                                             reduce_range=reduce_range)
        self.ch_axis = ch_axis
        self.min_vals = None
        self.max_vals = None
        if (
            self.qscheme == torch.per_channel_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric quantization for quint8"
            )

    def forward(self, x_orig):
        return self._forward(x_orig)

    @torch.jit.ignore
    def _forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        min_vals = self.min_vals
        max_vals = self.max_vals
        x_dim = x.size()

        new_axis_list = list(range(len(x_dim)))
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(tuple(new_axis_list))
        y = torch.flatten(y, start_dim=1)
        if min_vals is None or max_vals is None:
            min_vals = torch.min(y, 1)[0]
            max_vals = torch.max(y, 1)[0]
        else:
            min_vals = torch.min(torch.min(y, 1)[0], min_vals)
            max_vals = torch.max(torch.max(y, 1)[0], max_vals)
        self.min_vals = min_vals
        self.max_vals = max_vals
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        return self._calculate_per_channel_qparams(self.min_vals, self.max_vals)

    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_vals, self.max_vals)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(CustomPerChannelMinMaxObserver, self)._save_to_state_dict(
            destination, prefix, keep_vars)
        destination[prefix + 'min_vals'] = self.min_vals
        destination[prefix + 'max_vals'] = self.max_vals

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.min_vals = state_dict.pop(prefix + 'min_vals')
        self.max_vals = state_dict.pop(prefix + 'max_vals')
        super(CustomPerChannelMinMaxObserver,
              self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                          missing_keys, unexpected_keys, error_msgs)


class CustomMovingAveragePerChannelMinMaxObserver(CustomPerChannelMinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit

    The quantization parameters are computed the same way as in
    :class:`~torch.quantization.observer.MovingAverageMinMaxObserver`, with the
    difference that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(self, averaging_constant=0.01, ch_axis=0, dtype=torch.quint8,
                 qscheme=torch.per_channel_affine, reduce_range=False):
        super(CustomMovingAveragePerChannelMinMaxObserver, self).__init__(
            ch_axis=ch_axis, dtype=dtype, qscheme=qscheme,
            reduce_range=reduce_range)
        self.averaging_constant = averaging_constant

    def forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        min_vals = self.min_vals
        max_vals = self.max_vals
        x_dim = x.size()

        new_axis_list = list(range(len(x_dim)))
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(tuple(new_axis_list))
        y = torch.flatten(y, start_dim=1)
        if min_vals is None or max_vals is None:
            min_vals = torch.min(y, 1)[0]
            max_vals = torch.max(y, 1)[0]
        else:
            min_vals = min_vals + self.averaging_constant * \
                (torch.min(y, 1)[0] - min_vals)
            max_vals = max_vals + self.averaging_constant * \
                (torch.max(y, 1)[0] - max_vals)
        self.min_vals = min_vals
        self.max_vals = max_vals
        return x_orig

class CustomFakeQuantize(FakeQuantize):
    def __init__(self, observer=CustomMovingAverageMinMaxObserver, quant_min=0, quant_max=255, **observer_kwargs):
        super(CustomFakeQuantize, self).__init__(observer, quant_min, quant_max, **observer_kwargs)
        self.activation_post_process.set_qmin_qmax(quant_min, quant_max) 
        self.observer_cls = observer

    def set_qmin_qmax(self, quant_min, quant_max):
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.activation_post_process = self.observer_cls(quant_min=quant_min, quant_max=quant_max) # need to re-initialize the observer

# based on 'fbgemm' in qconfig
weight_fake_quant_dummy =\
    CustomFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                           quant_min=-128,
                           quant_max=127,
                           dtype=torch.qint8,
                           qscheme=torch.per_channel_symmetric,
                           reduce_range=False,
                           ch_axis=0)
disable_fake_quant(weight_fake_quant_dummy)

def get_activation_quant(nlevel: int, enable: bool=True):
    quant = CustomFakeQuantize.with_args(
        observer=CustomMovingAverageMinMaxObserver,
        dtype=torch.qint32,
        quant_min=0,
        quant_max=nlevel - 1,
        reduce_range=False)() # must call the builder to build an instance
    if not enable:
        disable_fake_quant(quant)
        # disable_observer(quant) # Do we disable observer or not?
    return quant

def get_qconfig(nlevel: int) -> QConfig:
    return QConfig(
        activation=get_activation_quant(nlevel),
        weight=weight_fake_quant_dummy)
