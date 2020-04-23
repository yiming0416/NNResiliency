import torch
from torch import nn
import torch.nn.functional as F 
import numpy as np
from typing import Callable
from cached_property import cached_property_with_ttl

# TODO: maybe NoisyModule is more generic?
class NoisyLayer(nn.Module):
    noisy = True
    # noise_range = 1
    noise_type_sampler_dict = {
        "gaussian": torch.randn,
        "uniform": lambda *args, **kwargs: 2 * torch.rand(args, **kwargs) - 1
    }
    _fixtest_flag = False # TODO: redundant
    sigma = None
    mu = 0
    pre_sample = False
    record_delta: bool = False

    # self.register_buffer('stdev')
    def __init__(self, *args,
                 use_range: bool=True,
                 pre_sample: bool=False,
                 match_range: bool=True,
                 **kwargs):
        if use_range is True:
            assert pre_sample is not True
        self.use_range = use_range
        self.match_range = match_range
        self.pre_sample = pre_sample
        self._noise_type = "gaussian"
        self.sampler = self.noise_type_sampler_dict[self._noise_type]
        self._noise_range = 1
        self.parameters_to_perturb = ["bias", "weight"]
        self.parameters_to_match = ["bias", "weight"]
        super().__init__(*args, **kwargs)

    # TODO: make it accept custom samplers
    @property
    def noise_type(self):
        return self._noise_type
    @noise_type.setter
    def noise_type(self, n_type: str):
        assert n_type in self.noise_type_sampler_dict, "Unknown noise_type"
        self._noise_type = n_type
        self.sampler = self.noise_type_sampler_dict[n_type]
    
    @property
    def noise_range(self):
        return self._noise_range
    @noise_range.setter
    def noise_range(self, noise_range: float):
        assert 0 <= noise_range and noise_range <= 1, "noise_range must be between [0,1]"
        self._noise_range = noise_range

    def sample_perturbation(self, param: torch.Tensor, num_samples: int, sampler: Callable=torch.randn) -> torch.Tensor:
        # TODO: for linear, if multiple samples, expand along axis 1; for conv2d, if multiple samples, expand along axis 1 as well.
        return sampler(param.size(), device=param.device)

    @property
    def fixtest_flag(self):
        return self._fixtest_flag
    @fixtest_flag.setter
    def fixtest_flag(self, fixtest_flag: bool):
        self._fixtest_flag = fixtest_flag
        if self._fixtest_flag:
            perturbation = self.get_perturbation()
            self.apply_perturbation(**perturbation)

    #@cached_property_with_ttl(ttl=0)
    @property
    def perturbation_stdev_dict(self) -> dict:
        epsilon = 1e-8
        stdev_dict = {}
        if not self.use_range:
            for param in self.parameters_to_perturb:
                if hasattr(self, param) and getattr(self, param) is not None:
                    stdev_dict[param] = 1
        else:
            for param in self.parameters_to_perturb: # bias and weight
                if hasattr(self, param) and getattr(self, param) is not None:
                    setattr(self, param + "_range", cal_range(getattr(self, param), self.noise_range))
                    stdev_dict[param] = max(getattr(self, param + "_range"), epsilon)
            if self.match_range: # only match bias and weight
                assert len(self.parameters_to_match) == 2, "Can only match ranges of 2 parameters"
                range_names_to_match = [param + "_range" for param in self.parameters_to_match]
                param_name_1, param_name_2 = self.parameters_to_match[0], self.parameters_to_match[1]
                if hasattr(self, range_names_to_match[0]) and hasattr(self, range_names_to_match[1]):
                    range_to_match = [getattr(self, rn) for rn in range_names_to_match]
                    if range_to_match[0] != 0 and range_to_match[1] != 0:
                        range_factor = range_to_match[0] / range_to_match[1]
                        self.merged_weight_range = cal_range(torch.cat((getattr(self, param_name_1).view(-1) / range_factor, getattr(self, param_name_2).view(-1))), self.noise_range)
                        stdev_dict[param_name_1] = self.merged_weight_range * range_factor # bias
                        stdev_dict[param_name_2] = self.merged_weight_range # weight
        return stdev_dict

    def get_perturbation(self) -> dict:
        """Return the parameter perturbation.
        self.record_delta: If self.record_delta = True, set the relevant registered_buffer with rescaled perturbation
        """
        perturbation_dict = {}
        # pylint: disable=unsubscriptable-object,not-an-iterable
        with torch.no_grad():
            for param in self.perturbation_stdev_dict:
                n = self.sample_perturbation(getattr(self, param), 1, sampler=self.sampler)
                perturbation_dict[param] = self.perturbation_stdev_dict[param] * self.sigma * n + self.mu * self.perturbation_stdev_dict[param]
                if self.record_delta:
                    assert self.noise_type == "gaussian", "delta values are only useful for Gaussian perturbation"
                    buffer_name = "delta_" + param
                    if not hasattr(self, buffer_name):
                        self.register_buffer(buffer_name, torch.empty_like(getattr(self, param)))
                    if self.mu is not 0:
                        # FIXME: what should this be?
                        self.__dict__[buffer_name] = n / self.perturbation_stdev_dict[param] - self.mu / self.sigma ** 2
                    else:
                        self.__dict__[buffer_name] = n / self.perturbation_stdev_dict[param]

        return perturbation_dict

    def apply_perturbation(self, **perturbation_dict):
        for param in perturbation_dict:
            perturbation = perturbation_dict[param]
            getattr(self, param).data.add_(perturbation)
        return

    def forward(self, input):
        raise NotImplementedError

class NoisyConv2d(NoisyLayer, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, mu=0, sigma=1, use_range=True, match_range=True):
        self.mu = mu
        self.sigma = sigma
        if bias:
            self.parameters_to_perturb = ["bias", "weight"]
        else:
            self.parameters_to_perturb = ["weight"]
            match_range = False

        super(NoisyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
              padding, dilation, groups, bias, use_range=use_range, match_range=match_range)

    def forward(self, input):
        if self.noisy and (self.sigma or self.mu) and (not self.fixtest_flag):
            perturbation_dict = self.get_perturbation()
            perturbed_weight = self.weight + perturbation_dict["weight"]
            perturbed_bias = self.bias + perturbation_dict["bias"] if self.bias is not None else None
            return F.conv2d(input, perturbed_weight, perturbed_bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
           ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.sigma:
            s += ', sigma={sigma}'
        return s.format(**self.__dict__)


class NoisyLinear(NoisyLayer, nn.Linear):
    def __init__(self, in_features, out_features, bias=True, mu=0, sigma=1, use_range=True, match_range=True):
        self.mu = mu
        self.sigma = sigma
        super(NoisyLinear, self).__init__(in_features, out_features, bias, use_range=use_range, match_range=match_range)

    def forward(self, input):
        if self.noisy and (self.sigma or self.mu) and (not self.fixtest_flag):
            perturbation_dict = self.get_perturbation()
            perturbed_weight = self.weight + perturbation_dict["weight"]
            perturbed_bias = self.bias + perturbation_dict["bias"] if self.bias is not None else None
            return F.linear(input, perturbed_weight, perturbed_bias)
        else:
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        s = super().extra_repr()
        if self.sigma:
            s += ', sigma={sigma}'
        return s.format(**self.__dict__)


# TODO: make this compatible with the get_perturbation & apply_perturbation
class NoisyIdentity(NoisyLayer, nn.Module):
    noisy = True
    def __init__(self, sigma=1):
        self.sigma = sigma
        self.fix_flag = False
        super(NoisyIdentity, self).__init__()

    def forward(self, x):
        if self.noisy and self.sigma and self.training:
            out = torch.ones_like(x) + torch.randn_like(x) * self.sigma
            return out * x

        elif self.noisy and self.sigma and (not self.training) and (not self.fix_flag):
            out = torch.ones_like(x) + torch.randn_like(x) * self.sigma
            self.register_buffer('out_fix', out)
            return out * x

        elif self.noisy and self.sigma and (not self.training) and self.fix_flag:
            if x.size() == self.out_fix.size():
                return self.out_fix * x
            else:
                return self.out_fix[:x.size(0)] * x                
        else:
            return x

class NoisyBN(NoisyLayer, nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, mu=0, sigma=1, use_range=True, match_range=True):
        self.mu = mu
        self.sigma = sigma
        self.num_features = num_features        
        # TODO: add noise to running_mean & running_var
        super(NoisyBN, self).__init__(num_features, eps, momentum, affine, track_running_stats, use_range=use_range, match_range=match_range)
        b_eff = torch.zeros(num_features)
        w_eff = torch.zeros(num_features, 1, 1, 1)
        if not hasattr(self, "b_eff"):
            self.register_buffer("b_eff", b_eff)
        else:
            self.b_eff = b_eff
        if not hasattr(self, "w_eff"):
            self.register_buffer("w_eff", w_eff)
        else:
            self.w_eff = w_eff
        self.parameters_to_match   = ["b_eff", "w_eff"]
        self.parameters_to_perturb = ["b_eff", "w_eff"]

    @property
    def fixtest_flag(self):
        return self._fixtest_flag
    @fixtest_flag.setter
    def fixtest_flag(self, fixtest_flag: bool):
        self._fixtest_flag = fixtest_flag
        if self._fixtest_flag:
            b_eff = self.bias - (self.running_mean * self.weight) / torch.sqrt(self.running_var + self.eps)
            w_eff = (self.weight / torch.sqrt(self.running_var + self.eps)).view(self.num_features,1,1,1)
            if not hasattr(self, "b_eff"):
                self.register_buffer("b_eff", b_eff)
            else:
                self.b_eff = b_eff
            if not hasattr(self, "w_eff"):
                self.register_buffer("w_eff", w_eff)
            else:
                self.w_eff = w_eff
            self.apply_perturbation(**self.get_perturbation())
     
    def forward(self, input):
        #if self.noisy and (self.sigma or self.mu) and (not self.fixtest_flag) and self.training:
        if not self.fixtest_flag: 
            bn_mean = input.mean(axis=(0,2,3))
            bn_var = input.var(axis=(0,2,3), unbiased=False)
            bn_weight, bn_bias = self.weight.detach(), self.bias.detach()
            self.b_eff = self.bias - (bn_mean * self.weight) / torch.sqrt(bn_var + self.eps)
            self.w_eff = (self.weight / torch.sqrt(bn_var + self.eps)).view(self.num_features,1,1,1)
            if self.noisy and (self.sigma or self.mu):
                perturbation_dict = self.get_perturbation()
                perturbed_w_eff = self.w_eff + perturbation_dict["w_eff"] if self.w_eff is not None else None
                perturbed_b_eff = self.b_eff + perturbation_dict["b_eff"] if self.b_eff is not None else None
                F.batch_norm(input, self.running_mean, self.running_var, bn_weight, bn_bias, self.training, self.momentum, self.eps)
                return F.conv2d(input, perturbed_w_eff, perturbed_b_eff, stride=1, groups=self.num_features)
            else:
                return F.conv2d(input, self.w_eff, self.b_eff, stride=1, groups=self.num_features)
        else:
            return F.conv2d(input, self.w_eff, self.b_eff, stride=1, groups=self.num_features)

def set_noisy(m, noisy=True):
    if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear) or isinstance(m, NoisyIdentity) or isinstance(m, NoisyBN):
        m.noisy = noisy

def set_clean(m):
    if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear) or isinstance(m, NoisyIdentity) or isinstance(m, NoisyBN):
        m.noisy = False

def set_gaussian_noise(m):
    if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear):
        m.noise_type = 'gaussian'

def set_uniform_noise(m):
    if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear):
        m.noise_type = 'uniform'

def set_noise_type(m, noise_type):
    if noise_type == "gaussian":
        set_gaussian_noise(m)
    elif noise_type == "uniform":
        set_uniform_noise(m)

def set_fixtest(m):
    if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear) or isinstance(m, NoisyBN):
        m.fixtest_flag = True

def set_noisyid_fix(m):
    if isinstance(m, NoisyIdentity):
        m.fix_flag = True

def set_noisyid_unfix(m):
    if isinstance(m, NoisyIdentity):
        m.fix_flag = False    

def set_grad_with_delta(m):
    if isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear) or isinstance(m, NoisyBN):
        if m.sigma is None or m.sigma == 0:
            return
        m.weight.grad = m.delta_weight
        if hasattr(m, "bias"):
            m.bias.grad = m.delta_bias


def cal_range(weight: torch.Tensor, noise_range_ratio: float):
    assert noise_range_ratio <= 1 and noise_range_ratio > 0
    # TODO: with torch.no_grad():
    if noise_range_ratio == 1:
        return (torch.max(weight) - torch.min(weight))/2

    top_rank = max(int(len(weight) * (1-noise_range_ratio)/2) + 1, 1)
    bottom_rank = min(len(weight) - top_rank + 1, len(weight))
    tmp = weight.view(-1)
    low_bound, _ = torch.kthvalue(tmp, top_rank)
    high_bound, _ = torch.kthvalue(tmp, bottom_rank)
    range_ = (high_bound - low_bound) / 2

    return range_