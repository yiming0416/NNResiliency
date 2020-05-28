import torch
from networks import *
from utils import *
import unittest
import argparse
import random
import numpy as np

class TestNoisyConv2dUnrolled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # fix random seeds
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # makeup a bunch of data
        cls.num_batches = 5
        cls.input_data = torch.randn(cls.num_batches, 4, 3, 32, 32)

        cls.layers, cls.ref_layers = cls.create_layers()
        cls.init_layers(cls.layers, cls.ref_layers)
    
    @classmethod
    def create_layers(cls):
        layers = []
        ref_layers = []

        layers.append(NoisyConv2dUnrolled(3, 2, (3,3), bias=True, stride=1, padding=1))
        ref_layers.append(nn.Conv2d(3, 2, (3,3), bias=True, stride=1, padding=1))

        layers.append(NoisyConv2dUnrolled(3, 2, (3,3), bias=True, stride=2, padding=1))
        ref_layers.append(nn.Conv2d(3, 2, (3,3), bias=True, stride=2, padding=1))

        layers.append(NoisyConv2dUnrolled(3, 2, (3,3), bias=True, stride=1, padding=5))
        ref_layers.append(nn.Conv2d(3, 2, (3,3), bias=True, stride=1, padding=5))

        layers.append(NoisyConv2dUnrolled(3, 2, (3,3), bias=True, stride=1, padding=1, dilation=2))
        ref_layers.append(nn.Conv2d(3, 2, (3,3), bias=True, stride=1, padding=1, dilation=2))

        # layer5 = NoisyConv2dUnrolled(3, 3, (3,3), bias=True, stride=1, padding=1, groups=3)
        # layer5_ref = nn.Conv2d(3, 3, (3,3), bias=True, stride=1, padding=1, groups=3)

        return layers, ref_layers
    
    @classmethod
    def init_layers(cls, layers, ref_layers):
        for i, (layer, layer_ref) in enumerate(zip(layers, ref_layers)):
            # print(i)
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.normal_(layer.bias)
            layer_ref.weight.data.copy_(layer.weight.data)
            layer_ref.bias.data.copy_(layer.bias.data)

    @classmethod
    def tearDownClass(cls):
        cls.num_batches = None
        cls.input_data = None
        cls.layers = None
        cls.ref_layers = None

    def setUp(self):
        self.layers, self.ref_layers = self.create_layers()
        self.init_layers(self.layers, self.ref_layers)

    def test_clean_forward(self):
        for layer, layer_ref in zip(self.layers, self.ref_layers):
            layer.noisy = False
            for data in self.input_data:
                out1 = layer(data)
                out2 = layer_ref(data)
                # print((out1 - out2).abs().max())
                idx = (out1 - out2).abs().argmax().item()
                msg = f"out1: {(out1.reshape(-1)[idx]).item()}\nout2: {(out2.reshape(-1)[idx]).item()}\ndiff.abs.max: {(out1 - out2).abs().max().item()}"
                self.assertTrue(torch.allclose(out1, out2, atol=1e-6), msg)

    # def test_noisy_forward(self):
    #     # TODO: set sigma to 0, mu to some number, see whether NoisyConv2d agrees with NoisyConv2dUnrolled
    #     pass

    # test first forward then fixtest
    def test_fixtest_clean1(self):
        for layer in self.layers:
            layer.sigma = 0
            layer(self.input_data[0])
            layer.fixtest_flag = True
        self.test_clean_forward()

    # test manually set output_size then fixtest
    def test_fixtest_clean2(self):
        output_size_list = []
        for layer in self.layers:
            output = layer(self.input_data[0])
            output_size_list.append(output.size()[-2:])

        self.setUp()
        for layer, output_size in zip(self.layers, output_size_list):
            layer.sigma = 0
            layer.output_size = output_size
            layer.fixtest_flag = True
        self.test_clean_forward()

class TestNoisyBNUnrolled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # fix random seeds
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # makeup a bunch of data
        cls.num_batches = 5
        cls.input_data = torch.randn(cls.num_batches, 4, 3, 32, 32)

        cls.layers, cls.ref_layers = cls.create_layers()
        cls.init_layers(cls.layers, cls.ref_layers)
    
    @classmethod
    def create_layers(cls):
        layers = []
        ref_layers = []

        layers.append(NoisyBNUnrolled(3))
        ref_layers.append(NoisyBN(3))

        return layers, ref_layers
    
    @classmethod
    def init_layers(cls, layers, ref_layers):
        for i, (layer, layer_ref) in enumerate(zip(layers, ref_layers)):
            # print(i)
            torch.nn.init.normal_(layer.weight)
            torch.nn.init.normal_(layer.bias)
            layer_ref.weight.data.copy_(layer.weight.data)
            layer_ref.bias.data.copy_(layer.bias.data)

    @classmethod
    def tearDownClass(cls):
        cls.num_batches = None
        cls.input_data = None
        cls.layers = None
        cls.ref_layers = None

    def setUp(self):
        self.layers, self.ref_layers = self.create_layers()
        self.init_layers(self.layers, self.ref_layers)

    def test_clean_forward(self):
        for layer, layer_ref in zip(self.layers, self.ref_layers):
            layer.noisy = False
            layer_ref.noisy = False
            layer.train()
            layer_ref.train()
            for data in self.input_data:
                out1 = layer(data)
                out2 = layer_ref(data)
                idx = (out1 - out2).abs().argmax().item()
                msg = f"out1: {(out1.reshape(-1)[idx]).item()}\nout2: {(out2.reshape(-1)[idx]).item()}\ndiff.abs.max: {(out1 - out2).abs().max().item()}"
                self.assertTrue(torch.allclose(out1, out2, atol=1e-6), msg)

    def test_noisy_forward(self):
        pass

    # test first forward then fixtest
    def test_fixtest_clean1(self):
        for layer in self.layers:
            layer.sigma = 0
            layer(self.input_data[0])
            layer.fixtest_flag = True
        for layer in self.ref_layers:
            layer.sigma = 0
            layer(self.input_data[0])
            layer.fixtest_flag = True
        self.test_clean_forward()

    # test manually set output_size then fixtest
    def test_fixtest_clean2(self):
        output_size_list = []
        for layer in self.layers:
            output = layer(self.input_data[0])
            output_size_list.append(output.size()[-2:])

        self.setUp()
        for layer, output_size in zip(self.layers, output_size_list):
            layer.sigma = 0
            layer.output_size = output_size
            layer.fixtest_flag = True
        for layer, output_size in zip(self.ref_layers, output_size_list):
            layer.sigma = 0
            layer.fixtest_flag = True
        self.test_clean_forward()

# class TestNoisyNet(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.num_batches = 5
#         args = argparse.Namespace(
#             net_type="resnet",
#             dataset="mnist",
#             training_noise_type="gaussian",
#             depth=18,
#             dropout_rate=0.3,
#             device=[0],
#             cpu=True
#         )
#         cls.dataset, _, cls.num_classes = getDatasets("mnist")
#         cls.dataloader = torch.utils.data.DataLoader(cls.dataset, batch_size=16, shuffle=True, num_workers=0)
#         cls.net, cls.net_name = getNetwork(args, cls.num_classes)
#         cls.use_cuda = torch.cuda.is_available() and not args.cpu
#         if cls.use_cuda:
#             if args.device:
#                 cls.device = torch.device('cuda:{:d}'.format(args.device[0]))
#             else:
#                 cls.device = torch.device('cuda')
#                 args.device = range(torch.cuda.device_count())
#         if cls.use_cuda:
#             if torch.cuda.device_count() > 1 and len(args.device) > 1:
#                 cls.net = torch.nn.DataParallel(cls.net, device_ids=range(torch.cuda.device_count()))
#             cls.net.cuda(device=cls.device)
#             # cudnn.benchmark = True

#     @classmethod
#     def tearDownClass(cls):
#         cls.dataset = None
#         cls.num_classes = None
#         cls.dataloader = None
#         cls.net = None
#         cls.net_name = None
#         cls.use_cuda = None
#         cls.device = None

#     def setUp(self):
#         pass

#     def test_forward(self):
#         # can be moved to a base class
#         # not urgent
#         pass

#     def test_fixtest(self):
#         pass

#     def test_noisy_forward(self):
#         ...

if __name__ == '__main__':
    unittest.main()