from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os, sys, time, datetime
import argparse
import random

import networks
from networks import *
from utils import *
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import json
import xlwt

from itertools import product
import pandas as pd
from trajectory import TrajectoryLogger, TrajectoryLog

parser = argparse.ArgumentParser(description='HA-SGD Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', default=0.01, type=float, help='learning_rate')
parser.add_argument('--num_epochs', '-n', default=200, type=int, help='num_epochs')
parser.add_argument('--epochs_lr_decay', '-a', default=60, type=int, help='epochs_for_lr_decay')
parser.add_argument('--lr_decay_rate', default=0.2, type=float, help='lr_decay_rate')
parser.add_argument('--batch_size', '-s', default=200, type=int, help='batch size')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=18, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout_rate', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'mnist'], help='dataset = [cifar10/cifar100/mnist]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--test_sample_num', type=int, default=20, help='The number of test runs per setting')
parser.add_argument('--load_model', type=str, default=None, help='Specify the model .pkl to load for testing/resuming training')
parser.add_argument('--training_noise', type=float, nargs='+', default=None, help='Set the training noise standard deviation')
parser.add_argument('--testing_noise', type=float, nargs='+', default=None, help='Set the testing noise standard deviation')
parser.add_argument('--testing_noise_mean_random_sign', action='store_true', help='Set the mean of the testing noise with random sign')
parser.add_argument('--training_noise_type', type=str, default='gaussian', choices=['gaussian', 'uniform'], help='noise_type = [gaussian/uniform]')
parser.add_argument('--testing_noise_type', type=str, default='gaussian', choices=['gaussian', 'uniform'], help='noise_type = [gaussian/uniform]')

#add new arguments to accept training_noise_mean and testing_noise_mean
parser.add_argument('--training_noise_mean', type=float, nargs='+', default=None, help='Set the mean of the training noise')
parser.add_argument('--testing_noise_mean', type=float, nargs='+', default=None, help='Set the mean of the testing noise')

parser.add_argument('--forward_samples', default=1, type=int, help='multi samples during forward')
parser.add_argument('--tensorboard', action='store_true', help='Turn on the tensorboard monitoring')
parser.add_argument('--regularization_type', type=str, choices=['l2', 'l1'], default='l2', help='Set the type of regularization')
parser.add_argument('--regularization', type=float, default=5e-4, help='Set the strength of regularization')
parser.add_argument('--seed', help='seed', type=int, default=42)
parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
parser.add_argument('--device', type=int, nargs='+', default=None, help='Set the device(s) to use')
parser.add_argument('--optim_type', default='SGD', type=str, choices=['SGD', 'EntropySGD', 'backpropless'], help='Set the type of optimizer')
parser.add_argument('--momentum', default=0.9, type=float, help='Set the momentum coefficient')
parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
parser.add_argument('--run_name', help='The name of this run (used for tensorboard)', type=str, default=None)

parser.add_argument('--test_with_std', action='store_true', help="fix mean, change std while testing")
parser.add_argument('--test_with_mean', action='store_true', help="fix std, change mean while testing")

parser.add_argument('--trajectory_dir', type=str, default=None, help='Set the directory for trajectory log')
parser.add_argument('--trajectory_interval', type=int, default=10, help='Set the interval of trajectory logging')
# parser.add_argument('--test_with_quantization', action='store_true', help="Test model with its quantized version")
parser.add_argument('--test_quantization_levels', type=int, nargs='+', default=None, help="The levels of quantization during testing")

if __name__ != "__main__":
    sys.exit(1)

args = parser.parse_args()

test_std_list = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
#test_mean_list = [0.0]
#test_mean_pos = [0.0]
#test_mean_neg = [0.0]

# test_mean_list = [-0.08, -0.06, -0.04, -0.02, -0.01, -0.004, 0.0, 0.004, 0.01, 0.02, 0.04 , 0.06, 0.08]
test_mean_list = [-0.004, 0.0, 0.004]
test_mean_pos = [0.0, 0.004, 0.01, 0.02, 0.04, 0.06, 0.08]
test_mean_neg = [-0.08, -0.06, -0.04, -0.02, -0.01, -0.004, 0.0]

# Set random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Set devices
device = torch.device('cpu')
use_cuda = torch.cuda.is_available() and not args.cpu
if use_cuda:
    if args.device:
        device = torch.device('cuda:{:d}'.format(args.device[0]))
    else:
        device = torch.device('cuda')
        args.device = range(torch.cuda.device_count())


###################################################
print('\n[Phase 1] : Data Preparation')
# start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, args.num_epochs, args.batch_size, args.optim_type
start_epoch, num_epochs, batch_size, optim_type = 0, args.num_epochs, args.batch_size, args.optim_type

trainset, testset, num_classes = getDatasets(args.dataset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

#####################################################
print('\n[Phase 2] : Model setup')
net, file_name = getNetwork(args, num_classes=num_classes)
print('| Building net...')
print (file_name)
net.apply(conv_init)
criterion = nn.CrossEntropyLoss()

if use_cuda:
    if torch.cuda.device_count() > 1 and len(args.device) > 1:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda(device=device)
    cudnn.benchmark = True

def train(net, epoch, optimizer, tensorboard_writer=None, clipper=None, trajectory_logger: TrajectoryLogger=None):
    net.train()
    net.apply(set_noisy)

    train_loss, acc, acc5 = AverageMeter(), AverageMeter(), AverageMeter()

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, optimizer.param_groups[0]['lr']))
        
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        global_step = epoch*len(trainloader) + batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(device=device), targets.cuda(device=device) # GPU settings
        inputs, targets = inputs, targets

        if args.forward_samples == 1:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            train_loss.update(loss.item(), inputs.size(0))
            loss.backward()
            optimizer.step()
            if trajectory_logger is not None:
                trajectory_logger.add_param_log(net, global_step)
                trajectory_logger.add_grad_log(net, global_step)
                trajectory_logger.commit()
        else:
            pass

        optimizer.step() # Optimizer update
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1,5))
        acc.update(prec1, targets.size(0))
        acc5.update(prec5, targets.size(0))

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [{:3d}/{:3d}] Iter[{:3d}/{:3d}]\t\tLoss: {:.4f} Acc@1: {:.3%}'.format(
            epoch, num_epochs, batch_idx+1, int(np.ceil(len(trainset)/batch_size)), train_loss.avg, acc.avg)
            )
        sys.stdout.flush()

        if type(tensorboard_writer) is SummaryWriter:
            tensorboard_writer.add_scalar("train_loss", loss.item(), global_step=global_step)
            tensorboard_writer.add_scalar("train_acc", prec1, global_step=global_step)
            tensorboard_writer.add_scalar("train_top5_acc", prec5, global_step=global_step)
            tensorboard_writer.flush()

    return acc.avg, acc5.avg, train_loss.avg


def test(net, epoch, dataloader):
    net.eval()
    test_loss, acc, acc5 = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if use_cuda:
                inputs, targets = inputs.cuda(device=device), targets.cuda(device=device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss.update(loss.item(), targets.size(0))
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1,5))
            acc.update(prec1, targets.size(0))
            acc5.update(prec5, targets.size(0))

    print("\n| Validation Epoch #{:d}\t\t\tLoss: {:.4f} Acc@1: {:.2%}".format(epoch, loss.item(), acc.avg))

    return acc.avg, acc5.avg


def prepare_network_perturbation(
        net, noise_type: str = 'gaussian', fixtest: bool = False,
        perturbation_level: float = None, perturbation_mean: float = None):
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
        qat: bool = False, num_calibration_batchs: int = 10):
        # The last two arguments are redundant for now
    if num_quantization_levels is None:
        return
    # Specify quantization configuration
    net.set_quantization_level(num_quantization_levels)
    net.enable_quantization()
    # Calibrate with the test set TODO: use the training set to calibrate
    test(net, -1, calibration_dataloader)
    print('Post Training Quantization: Calibration done')
    net.apply(disable_observer)
    # net.qconfig = get_qconfig(num_quantization_levels)
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

def test_with_std_mean(net, checkpoint, epoch = 0, test_mean_list=[None],
                       test_std_list=[None], test_quantization_levels=[None],
                       sample_num=1, writer=None):
    if test_std_list is None:
        test_std_list = [None]
    if test_mean_list is None:
        test_mean_list = [None]
    if test_quantization_levels is None:
        test_quantization_levels = [None]
    results = []
    for stdev, mean, quant_levels in product(test_std_list, test_mean_list, test_quantization_levels):
        def prepare_and_test():
            net.load_state_dict(checkpoint['state_dict'], strict=False)
            if use_cuda:
                net.to(device)
            else:
                net.to("cpu")
            net.eval()
            net.apply(set_noisy)
            prepare_network_perturbation(net=net, noise_type=args.testing_noise_type, fixtest=True, perturbation_level=stdev, perturbation_mean=mean)
            prepare_network_quantization(net=net, num_quantization_levels=quant_levels, calibration_dataloader=trainloader, qat=False)
            test_acc, test_acc_5 = test(net, epoch, testloader)
            if args.tensorboard and writer is not None:
                # TODO: the proper value of the global_step?
                writer.add_scalar(f"test_acc/{mean}", test_acc, global_step=(epoch+1)*len(trainloader))
                writer.add_scalar(f"test_top5_acc/{mean}", test_acc_5, global_step=(epoch+1)*len(trainloader))
            return test_acc.cpu().item(), test_acc_5.cpu().item()

        print (f'test noise stdev: {stdev}, test noise mean: {mean},'
               f' test quant levels: {quant_levels}')
        acc_tuple_list = [prepare_and_test() for i in range(sample_num)]
        test_acc_list, test_acc5_list = zip(*acc_tuple_list)
        results.append({
            "stdev": stdev, "mean": mean, "quant_levels": quant_levels,
            "test_acc": test_acc_list, 
            "test_acc5": test_acc5_list, 
        })
    df = pd.DataFrame(results)
    df["test_acc_avg"] = df["test_acc"].apply(np.mean)
    df["test_acc5_avg"] = df["test_acc5"].apply(np.mean)
    df = df.fillna(0)
    return df

def save_model(net, save_point, args, metric = 1, stats_dict: dict=None):
    state = {
            'state_dict': net.state_dict(),
            'args': args
    }
    if stats_dict is not None:
        state.update(stats_dict)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(save_point):
        os.mkdir(save_point)
    if metric==1:
        save_file = os.path.join(save_point, file_name + "_metric1.pkl")
        torch.save(state, save_file)
        print(f'| Saved Best model to \n{save_file}\nstats = {stats_dict}')
    elif metric==2:
        save_file = os.path.join(save_point, file_name + "_metric2.pkl")
        torch.save(state, save_file)
        print(f'| Saved Best model to \n{save_file}\nstats = {stats_dict}')
        
    save_file = os.path.join(save_point, file_name + "_current.pkl")
    torch.save(state, save_file)
    print(f'| Saved Current model to \n {save_file}')

#######################################################
if args.testOnly:
    print ('\n Test Only Mode')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    if args.load_model:
        checkpoint_file = args.load_model
    else:
        checkpoint_file = './checkpoint/'+args.dataset+'/'+args.training_noise_type+'/'+file_name + '_metric1.pkl'
    checkpoint = torch.load(checkpoint_file)

    test_acc_df = test_with_std_mean(
        net, checkpoint, test_mean_list=args.testing_noise_mean,
        test_std_list=args.testing_noise,
        test_quantization_levels=args.test_quantization_levels,
        sample_num=args.test_sample_num
    )

    with open(os.path.join('test', file_name + '_metric1.test'), 'a') as f:
        f.write('\n')
        json.dump({
            "args": vars(args),
            "test_acc_df": test_acc_df.to_json() # load with `pd.read_json()`
        },f)

    sys.exit(0)

######################################################
print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.regularization, nesterov=args.nesterov)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs_lr_decay, gamma=args.lr_decay_rate)

writer = None
if args.tensorboard:
    now = datetime.datetime.now()
    log_identifier = file_name + '_' + now.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("tensorboard_dir/new_BN", log_identifier)
    writer = SummaryWriter(log_dir=log_dir)
    print("| Tensorboard record: " + log_identifier)
    writer.add_text("run-args", args.__repr__(), global_step=None, walltime=None)

best_acc_1 = 0
best_acc_2 = 0
elapsed_time = 0
save_point = os.path.join('checkpoint', args.dataset, args.training_noise_type)

if args.trajectory_dir is not None:
    trajectory_logger = TrajectoryLogger(net, args.trajectory_dir, args.trajectory_interval)
else:
    trajectory_logger = None

for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()
    # train
    if args.optim_type == "SGD":
        prepare_network_perturbation(
            net, noise_type=args.training_noise_type, fixtest=False,
            perturbation_level=args.training_noise, perturbation_mean=args.training_noise_mean
        )

        train_acc, train_acc_5, train_loss = train(net, epoch, optimizer, tensorboard_writer=writer, trajectory_logger=trajectory_logger)
        scheduler.step()
    elif args.optim_type == "EntropySGD":
        pass
    elif args.optim_type == "backpropless":
        pass
    save_model(net, save_point, args, 0)
    # test
    net_test, _ = getNetwork(args, num_classes)
    net_test.to(device)
    checkpoint_file = './checkpoint/'+args.dataset+'/'+args.training_noise_type+'/'+file_name + '_current.pkl'
    checkpoint = torch.load(checkpoint_file)
    test_acc_df = test_with_std_mean(net_test, checkpoint, epoch = epoch, test_std_list=args.testing_noise, test_mean_list=args.testing_noise_mean, sample_num=1, writer=writer)

    # TODO: not dealing with training & testing quant_level yet
    training_noise_stdev = args.training_noise[0] if args.training_noise is not None else 0
    training_noise_mean = args.training_noise_mean[0] if args.training_noise_mean is not None else 0
    metric_1 = test_acc_df[
        (test_acc_df['mean'] == training_noise_mean) # & (test_acc_df['stdev'] == training_noise_stdev)
    ]["test_acc_avg"]
    assert len(metric_1) > 0, "No metric1 because not testing for the training case"
    best_metric_1 = metric_1.values[0]

    if best_metric_1 > best_acc_1:
        print(best_metric_1)
        save_model(net, save_point, args, 1, {"acc": best_metric_1, "epoch": epoch})
        best_acc_1 = best_metric_1

    # if args.training_noise_mean is not None:       
    #     if args.training_noise_mean[0] > 0:
    #         best_metric_2 = sum(test_acc_dict[i] for i in test_mean_pos) / len(test_mean_pos)
    #     elif args.training_noise_mean[0] < 0:
    #         best_metric_2 = sum(test_acc_dict[i] for i in test_mean_neg) / len(test_mean_neg)
    # else:
    #     best_metric_2 = test_acc_dict[0.0]

    # if best_metric_2 > best_acc_2:
    #     print (best_metric_2)
    #     save_model(net, save_point, args, 2, {"acc": best_metric_2, "epoch": epoch})
    #     best_acc_2 = best_metric_2

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))
    if type(writer) is SummaryWriter:
        writer.flush()

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = {:.2%}'.format(best_acc_1))
print('* Test results : Acc@1 = {:.2%}'.format(best_acc_2))
