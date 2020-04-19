import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .noisy_layers import *
from .network_utils import children_of_class
from typing import Iterable
from itertools import cycle

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)

class LeNet(nn.Module):
    def __init__(self, num_classes, sigma_list=None, input_size=28, input_channel=1):
        super(LeNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = NoisyConv2d(input_channel, 6, 5)
        self.conv2 = NoisyConv2d(6, 16, 5)
        self.fc1   = NoisyLinear(16 * (input_size // 4 - 3)  ** 2, 120)
        self.fc2   = NoisyLinear(120, 84)
        self.fc3   = NoisyLinear(84, num_classes)
        self.set_sigma_list(sigma_list)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return(out)

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

    def set_sigma_list(self, sigma_list):
        """ Allow None, scalar, 1-length list or
        list with the length of the number of noisy layers
        """
        noisy_layers = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]
        if sigma_list is None:
            sigma_list = [sigma_list] * len(noisy_layers)
        else:
            try:
                sigma_list = float(sigma_list)
                sigma_list = [sigma_list] * len(noisy_layers)
            except:
                assert type(sigma_list) == list
                if len(sigma_list) == 1:
                    sigma_list = sigma_list * len(noisy_layers)
                else:
                    assert len(sigma_list) == len(noisy_layers)
        for m, s in zip(noisy_layers, sigma_list):
            m.sigma = s
