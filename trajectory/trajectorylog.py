import torch
import numpy as np
import os
from utils import create_or_clear_dir
import re
from collections import OrderedDict

from typing import Iterable

class TrajectoryLog():
    def __init__(self, log_dir, log_type, device="cpu"):
        self.log_type = log_type
        self.log_dir = log_dir
        if not os.path.isdir(self.log_dir):
            raise Exception("Log dir {} doesn't exist".format(self.log_dir))
        self.index = self._make_index(self.log_dir, os.listdir(self.log_dir))
        self.device = device

    def __getitem__(self, i):
        keys = None
        def extract_value(file):
            nonlocal keys
            dictionary = torch.load(file, map_location=self.device)
            if keys is None:
                keys = dictionary.keys()
            value = torch.cat(
                [torch.stack([v.view(-1).float() for v in dictionary[k]]) # (# obs x len(param_k))
                 for k in keys], dim=1
            ) # (# obs x len(all_params))
            return value
        if isinstance(i, slice):
            values = []
            files = self.index[i]
            for file in files:
                values.append(extract_value(file))
            return values
        elif isinstance(i, int):
            file = self.index[i]
            return extract_value(file)
        else:
            raise TypeError("Invalid argument type.")
            
    def __len__(self):
        return len(self.index)

    def _make_index(self, base:str, filename_list: Iterable) -> np.ndarray:
        regex = re.compile(r"^(\d+)\." + self.log_type + "$")
        index = []
        key = []
        for filename in filename_list:
            # fn = os.path.basename(filename)
            match = regex.match(filename)
            if match is None:
                continue
            else:
                index.append(os.path.join(base, filename))
                key.append(int(match.group(1)))
        return np.array(index)[np.argsort(key)]

def load_net_from_param_log(net: torch.nn.Module, param_log_path: str):
    param_dict = torch.load(param_log_path)
    for name, param in net.named_parameters():
        param_data = param_dict[name][0]
        param.data.copy_(param_data)

def set_model_params_from_vec(net: torch.nn.Module, param_vector: torch.Tensor):
    idx = 0
    for name, param in net.named_parameters():
        nelement = param.nelement()
        param.data.copy_(param_vector[idx:idx+nelement].view(param.size()))
        idx += nelement

class TrajectoryLogger():
    def __init__(self, net: torch.nn.Module, log_dir: str, log_interval: int=1):
        self.log_dir = log_dir
        self.log_interval = log_interval
        create_or_clear_dir(self.log_dir)

        self.param_names = [param_name for param_name, _ in net.named_parameters()]
        self.param_buffer = {}
        self.grad_buffer = {}

    def _get_step_filename(self, global_step: int, type_: str):
        return os.path.join(self.log_dir, f"{global_step}.{type_}")
    
    def _load_param_log(self, global_step: int):
        return torch.load(self._get_step_filename(global_step, "param"))
    def _load_grad_log(self, global_step: int):
        return torch.load(self._get_step_filename(global_step, "grad"))

    def add_param_log(self, net: torch.nn.Module, global_step: int) -> None:
        """Add the parameters of the net to buffer
        """
        if global_step % self.log_interval != 0:
            return
        if global_step in self.param_buffer:
            if self.param_buffer[global_step] is None: # was committed before
                # load buffer from disk
                self.param_buffer = self._load_param_log(global_step)
        else:
            # create new empty buffer 
            # self.param_buffer[global_step] = {name: [] for name, _ in net.named_parameters()}, 
            self.param_buffer[global_step] = OrderedDict.fromkeys(self.param_names)
            for name in self.param_buffer[global_step]:
                self.param_buffer[global_step][name] = []
        # append current log
        for name, param in net.named_parameters():
            assert name in self.param_buffer[global_step]
            self.param_buffer[global_step][name].append(param.clone().detach())

    def add_grad_log(self, net: torch.nn.Module, global_step) -> None:
        """Add the gradients of the net to buffer
        """
        if global_step % self.log_interval != 0:
            return
        if global_step in self.grad_buffer:
            if self.grad_buffer[global_step] is None: # was committed before
                # load buffer from disk
                self.grad_buffer = self._load_grad_log(global_step)
        else:
            # create new empty buffer 
            # self.grad_buffer[global_step] = {name: [] for name, _ in net.named_parameters()}, 
            self.grad_buffer[global_step] = OrderedDict.fromkeys(self.param_names)
            for name in self.param_names:
                self.grad_buffer[global_step][name] = []
        # append current log
        for name, param in net.named_parameters():
            assert name in self.grad_buffer[global_step]
            self.grad_buffer[global_step][name].append(param.grad.clone().detach())

    def commit(self) -> None:
        """Commit the buffer to disk
        """
        for step in self.param_buffer:
            if self.param_buffer[step] is not None:
                torch.save(self.param_buffer[step], self._get_step_filename(step, "param"))
                self.param_buffer[step] = None

        for step in self.grad_buffer:
            if self.grad_buffer[step] is not None:
                torch.save(self.grad_buffer[step], self._get_step_filename(step, "grad"))
                self.grad_buffer[step] = None

    def read_param_log(self):
        return TrajectoryLog(self.log_dir, log_type="param")

    def read_grad_log(self):
        return TrajectoryLog(self.log_dir, log_type="grad")