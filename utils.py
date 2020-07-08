from networks import *
import config as cf
import torchvision
import torchvision.transforms as transforms
import sys
import os, shutil

def getDatasets(dataset: str):
    transform_train_CIFAR = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
    ]) # meanstd transformation

    transform_train_MNIST = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
    ]) # meanstd transformation

    transform_test_CIFAR = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
    ])

    transform_test_MNIST = transforms.Compose([
        # transforms.Pad(padding=2, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
    ])
    if(dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_CIFAR)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_CIFAR)
        num_classes = 10
    elif(dataset == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_CIFAR)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test_CIFAR)
        num_classes = 100
    elif(dataset == 'mnist'):
        print("| Preparing MNIST dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train_MNIST)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test_MNIST)
        num_classes = 10
    return trainset, testset, num_classes

def getNetworkName(args):
    parts = []
    parts.append(args.net_type)
    if args.training_noise_type == 'gaussian' and args.training_noise is None:
        parts.append('noise_std-[0.0]')
    elif args.training_noise_type == 'uniform':
        parts.append('noise-uniform')
    else:
        parts.append('noise_std-{}'.format(args.training_noise))
    if args.training_noise_mean is None:
        parts.append('noise_mean-[0.0]')
    else:
        parts.append('noise_mean-{}'.format(args.training_noise_mean))

    parts.append('{}-{}'.format(args.regularization_type, args.regularization))
    parts.append('dropout-{}'.format(args.dropout_rate))
    parts.append('lr-{}'.format(args.lr))
    parts.append('epochs-{}'.format(args.num_epochs))
    parts.append('lrdecayepoch-{}'.format(args.epochs_lr_decay))
    if args.forward_samples != 1:
        parts.append('forward-{}'.format(args.forward_samples))
    if args.optim_type == "EntropySGD":
        parts.append('entropySGD')
    if args.run_name:
        parts.append(args.run_name)

    file_name = '_'.join(parts)
    return file_name


# Return network & file name
def _get_network(net_type: str, depth: int, dropout_rate: float, dataset: str, num_classes: int,
                 widen_factor: int=1, training_noise_type: str = "gaussian", training_noise: float = None):
    if net_type == 'lenet':
        if dataset == 'mnist':
            net = LeNet(num_classes, input_size=28, input_channel=1)
        if dataset == 'cifar10':
            net = LeNet(num_classes, input_size=32, input_channel=3)
    if net_type == 'resnet':
        if dataset == 'mnist':
            net = ResNet(depth, num_classes, use_dropout = True, dropout_rate = dropout_rate, in_channel=1)
        else:
            net = ResNet(depth, num_classes, use_dropout = True, dropout_rate = dropout_rate, in_channel=3)
    if net_type == 'wide_resnet':
        net = WideResNet(depth, widen_factor, dropout_rate, num_classes)

    if training_noise_type == 'gaussian' and training_noise is None:
        net.apply(set_gaussian_noise)
    elif training_noise_type == 'uniform':
        net.apply(set_uniform_noise)
    else:
        net.apply(set_gaussian_noise)
    return net

def getNetwork(args, num_classes: int):
    net = _get_network(net_type=args.net_type, depth=args.depth, dropout_rate=args.dropout_rate,
                       dataset=args.dataset, num_classes=num_classes, widen_factor=args.widen_factor,
                       training_noise_type=args.training_noise_type, training_noise=args.training_noise)
    file_name = getNetworkName(args)
    return net, file_name

def create_or_clear_dir(path, force=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            # try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            # except Exception as e:
                # print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        if force:
            os.unlink(path)
            os.makedirs(path)
        else:
            raise NotADirectoryError("f{path} is a file")
