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
parser.add_argument('--device', type=int, nargs='+', default=None, help='Set the device(s) to use')
parser.add_argument('--optim_type', default='SGD', type=str, choices=['SGD', 'EntropySGD', 'backpropless'], help='Set the type of optimizer')
parser.add_argument('--momentum', default=0.9, type=float, help='Set the momentum coefficient')
parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
parser.add_argument('--run_name', help='The name of this run (used for tensorboard)', type=str, default=None)

parser.add_argument('--test_with_std', action='store_true', help="fix mean, change std while testing")
parser.add_argument('--test_with_mean', action='store_true', help="fix std, change mean while testing")

if __name__ != "__main__":
    sys.exit(1)

args = parser.parse_args()

test_std_list = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
test_mean_list = [-0.04, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04]
test_mean_pos = [0.0, 0.005, 0.01, 0.02, 0.04]
test_mean_neg = [-0.04, -0.02, -0.01, -0.005, 0.0]

# Set random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Set devices
device = torch.device('cpu')
use_cuda = torch.cuda.is_available()
if use_cuda:
    if args.device:
        device = torch.device('cuda:{:d}'.format(args.device[0]))
    else:
        device = torch.device('cuda')
        args.device = range(torch.cuda.device_count())


###################################################
print('\n[Phase 1] : Data Preparation')
transform_train_CIFAR = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_train_MNIST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test_CIFAR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

transform_test_MNIST = transforms.Compose([
    # transforms.Pad(padding=2, fill=0),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, args.num_epochs, args.batch_size, args.optim_type

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_CIFAR)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_CIFAR)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_CIFAR)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test_CIFAR)
    num_classes = 100
elif(args.dataset == 'mnist'):
    print("| Preparing MNIST dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train_MNIST)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test_MNIST)
    num_classes = 10

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


def train(net, epoch, optimizer, perturbation_level=None, tensorboard_writer=None, clipper=None):
    net.train()
    net.apply(set_noisy)
    if perturbation_level is None:
        perturbation_level = args.training_noise
    
    if args.training_noise_type == 'gaussian':
        net.apply(set_gaussian_noise)
        if isinstance(net, nn.DataParallel):
            net.module.set_sigma_list(perturbation_level)
            net.module.set_mu_list(args.training_noise_mean)
        else:
            net.set_sigma_list(perturbation_level)
            net.set_mu_list(args.training_noise_mean)

    elif args.training_noise_type == 'uniform':
        net.apply(set_uniform_noise)
        if isinstance(net, nn.DataParallel):
            net.module.set_sigma_list(1)
        else:
            net.set_sigma_list(1)

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


def test(net, epoch, test_std, test_mean, tensorboard_writer=None):
    net.eval()
    if isinstance(net, nn.DataParallel):
        net.module.set_sigma_list(test_std)
        net.module.set_mu_list(test_mean)
    else:
        net.set_sigma_list(test_std)
        net.set_mu_list(test_mean)
    net.apply(set_fixtest)

    test_loss, acc, acc5 = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
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
    else:
        save_file = os.path.join(save_point, file_name + "_metric2.pkl")
    torch.save(state, save_file)
    print(f'| Saved Best model to \n{save_file}\nstats = {stats_dict}')

#######################################################
if args.testOnly:
    print ('\n Test Only Mode')
    test_acc_avg = []
    test_acc_5_avg = []
    test_acc_all = {}
    test_acc_5_all = {}
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'

    if args.load_model:
        checkpoint_file = args.load_model
    else:
        checkpoint_file = './checkpoint/'+args.dataset+'/'+args.training_noise_type+'/'+file_name + '_metric1.pkl'
    checkpoint = torch.load(checkpoint_file)
    samples = 20

    for test_mean in test_mean_list:
        print ('test noise mean:{}'.format(test_mean))
        test_acc_all[str(test_mean)] = []
        test_acc_5_all[str(test_mean)] = []
        acc_avg = 0.0
        acc_5_avg = 0.0
        for i in range(samples):
            net.load_state_dict(checkpoint['state_dict'], strict=False)
            if use_cuda:
                net.to(device)        
            if args.testing_noise_type == 'gaussian':
                net.apply(set_gaussian_noise)
            elif args.testing_noise_type == 'uniform':
                net.apply(set_uniform_noise)
            net.eval()
            net.apply(set_noisy)
            test_acc, test_acc_5 = test(net, 0, args.training_noise, test_mean)
            test_acc_all[str(test_mean)].append(test_acc.cpu().item())
            test_acc_5_all[str(test_mean)].append(test_acc_5.cpu().item())
            acc_avg += test_acc/samples
            acc_5_avg += test_acc_5/samples

        test_acc_avg.append(acc_avg.cpu().item())
        test_acc_5_avg.append(acc_5_avg.cpu().item())

    with open(os.path.join('test', file_name + '_metric1.test'), 'a') as f:
        f.write('\n')
        json.dump({
            "args": vars(args),
            "test_mean_list": test_mean_list,
            "test_acc_avg": test_acc_avg,
            "test_acc_5_avg": test_acc_5_avg,
            "test_acc_all": test_acc_all,
            "test_acc_5_all": test_acc_all            
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
    log_dir = os.path.join("tensorboard_dir", log_identifier)
    writer = SummaryWriter(log_dir=log_dir)
    print("| Tensorboard record: " + log_identifier)
    writer.add_text("run-args", args.__repr__(), global_step=None, walltime=None)

best_acc_1 = 0
best_acc_2 = 0
elapsed_time = 0
save_point = os.path.join('checkpoint', args.dataset, args.training_noise_type)

for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()
    # train
    if args.optim_type == "SGD": 
        train_acc, train_acc_5, train_loss = train(net, epoch, optimizer, tensorboard_writer=writer)
        scheduler.step()
    elif args.optim_type == "EntropySGD":
        pass
    elif args.optim_type == "backpropless":
        pass

    # test
    '''
    if args.testing_noise:
        # clean test
        net.apply(set_clean)
        test_acc, test_acc_5 = test(net, epoch, tensorboard_writer=writer)
        if args.tensorboard:
            writer.add_scalar("clean_test_acc", test_acc, global_step=(epoch+1)*len(trainloader))
            writer.add_scalar("clean_test_top5_acc", test_acc_5, global_step=(epoch+1)*len(trainloader))
        # noisy test
        net.apply(set_noisy)
        test_acc, test_acc_5 = test(net, epoch, tensorboard_writer=writer)
        if args.tensorboard:
            writer.add_scalar("noisy_test_acc", test_acc, global_step=(epoch+1)*len(trainloader))
            writer.add_scalar("noisy_test_top5_acc", test_acc_5, global_step=(epoch+1)*len(trainloader))
    '''
    
    test_acc_dict = {}
    net_test, _, = getNetwork(args, num_classes=num_classes)
    net_test.load_state_dict(net.state_dict())
    net_test.to(device)
    if args.test_with_std:
        for test_std in test_std_list:
            print('\n test noise std:{}'.format(test_std))
            print('test noise mean:{}'.format(args.training_noise_mean))
            net_test.apply(set_noisy)
            test_acc, test_acc_5 = test(net_test, epoch, test_std, args.training_noise_mean, writer)
            test_acc_dict[test_std] = test_acc
            if args.tensorboard:
                writer.add_scalar(f"test_acc/{test_std}", test_acc, global_step=(epoch+1)*len(trainloader))
                writer.add_scalar(f"test_top5_acc/{test_std}", test_acc_5, global_step=(epoch+1)*len(trainloader))

    elif args.test_with_mean:
        for test_mean in test_mean_list:
            print('\n test noise std:{}'.format(args.training_noise))
            print('test noise mean:{}'.format(test_mean))

            net_test.apply(set_noisy)
            test_acc, test_acc_5 = test(net_test, epoch, args.training_noise, test_mean, writer)
            test_acc_dict[test_mean] = test_acc
            if args.tensorboard:
                writer.add_scalar(f"test_acc/{test_mean}", test_acc, global_step=(epoch+1)*len(trainloader))
                writer.add_scalar(f"test_top5_acc/{test_mean}", test_acc_5, global_step=(epoch+1)*len(trainloader))
            
        if args.training_noise_mean is not None:
            best_metric_1 = test_acc_dict[args.training_noise_mean[0]]
        else:
            best_metric_1 = test_acc_dict[0.0]
        
        if args.training_noise_mean is not None:       
            if args.training_noise_mean[0] > 0:
                best_metric_2 = sum(test_acc_dict[i] for i in test_mean_pos) / len(test_mean_pos)
            elif args.training_noise_mean[0] < 0:
                best_metric_2 = sum(test_acc_dict[i] for i in test_mean_neg) / len(test_mean_neg)
        else:
            best_metric_2 = test_acc_dict[0.0]
    

    if best_metric_1 > best_acc_1:
        print (best_metric_1)
        save_model(net, save_point, args, 1, {"acc": best_metric_1, "epoch": epoch})
        best_acc_1 = best_metric_1

    if best_metric_2 > best_acc_2:
        print (best_metric_2)
        save_model(net, save_point, args, 2, {"acc": best_metric_2, "epoch": epoch})
        best_acc_2 = best_metric_2

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))
    if type(writer) is SummaryWriter:
        writer.flush()


print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = {:.2%}'.format(best_acc_1))
print('* Test results : Acc@1 = {:.2%}'.format(best_acc_2))