import torch
from networks import *
from utils import *
import unittest
import argparse
import random
import numpy as np

from training_functions import quantize_network
from trajectory.utils import extract_param_vec

class TestQuantizeNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_batches = 5
        args = argparse.Namespace(
            net_type="lenet",
            dataset="mnist",
            training_noise_type="gaussian",
            depth=18,
            dropout_rate=0.3,
            device=[0],
            cpu=True
        )
        cls.dataset, _, cls.num_classes = getDatasets("mnist")
        cls.dataloader = torch.utils.data.DataLoader(cls.dataset, batch_size=16, shuffle=True, num_workers=0)

    def test_quantize_network(self):
        for nlevels in [2,3,4,5,6,7,8,10,16,32,64,96,128,256]:
            self.network = LeNet(10)
            # self.network.to('cpu')
            self.network.eval()
            quantize_network(self.network, num_quantization_levels=nlevels, calibration_dataloader=self.dataloader)
            for name, child in self.network.named_children():
                print(f"{name}: {type(child)}")
                # print(child.qconfig)
                # try:
                #     weight_levels = torch.tensor([l.unique().nelement() for l in child.weight().dequantize().detach()])
                #     bias_levels = child.bias().detach().unique().nelement()
                #     print(f"weight levels: {weight_levels}, set levels: {nlevels}")
                #     self.assertTrue(torch.all(weight_levels <= nlevels), name + " weight")
                #     print(f"bias levels: {bias_levels}, set levels: {nlevels}")
                #     self.assertLessEqual(bias_levels, nlevels, name + " bias")
                # except Exception as e:
                # print(e)
                if hasattr(child, "weight") and child.weight is not None:
                    weight_levels = torch.tensor([l.unique().nelement() for l in child.weight.detach()])
                    print(f"weight levels: {weight_levels}, set levels: {nlevels}")
                    self.assertTrue(torch.all(weight_levels <= nlevels), name + " weight")
                if hasattr(child, "bias") and child.bias is not None:
                    bias_levels = child.bias.detach().unique().nelement()
                    print(f"bias levels: {bias_levels}, set levels: {nlevels}")
                    self.assertLessEqual(bias_levels, nlevels, name + " bias")


if __name__ == '__main__':
    unittest.main()