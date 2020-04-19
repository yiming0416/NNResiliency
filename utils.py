from networks import *

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
        parts.append('noise_mean-[0.0]'.format(args.training_noise))
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
def getNetwork(args, num_classes):
    if args.net_type == 'lenet':
        if args.dataset == 'mnist':
            net = LeNet(num_classes, input_size=28, input_channel=1)
        if args.dataset == 'cifar10':
            net = LeNet(num_classes, input_size=32, input_channel=3)
    if args.net_type == 'resnet':
        if args.dataset == 'mnist':
            net = ResNet(args.depth, num_classes, use_dropout = True, dropout_rate = args.dropout_rate, in_channel=1)
        else:
            net = ResNet(args.depth, num_classes, use_dropout = True, dropout_rate = args.dropout_rate, in_channel=3)

    if args.training_noise_type == 'gaussian' and args.training_noise is None:
        net.apply(set_gaussian_noise)
    elif args.training_noise_type == 'uniform':
        net.apply(set_uniform_noise)
    else:
        net.apply(set_gaussian_noise)

    file_name = getNetworkName(args)

    return net, file_name


def write_acc_excel(args, book, testing_noise_list, noisy_test_acc, noisy_test_acc_5, noisy_test_acc_all, noisy_test_acc_5_all):
    row, col = 0, 0
    _, file_name = getNetwork(args)
    sheet = book.add_sheet('1')
        
    for i in range (len(testing_noise_list)):
        sheet.write(row, col+i+1, testing_noise_list[i])

    row += 1
    sheet.write(row, col, file_name+'_top1')
    for i in range (len(noisy_test_acc)):
        sheet.write(row, col+i+1, noisy_test_acc[i])

    row += 1
    for i in range(len(testing_noise_list)):
        col += 1
        for j in range(len(noisy_test_acc_all[str(testing_noise_list[i])])):
            sheet.write(row+j+1, col, noisy_test_acc_all[str(testing_noise_list[i])][j])


    row += (len(noisy_test_acc_all['0.02'])+1)
    col = 0
    for i in range(len(testing_noise_list)):
        sheet.write(row, col+i+1, np.std(np.asarray(noisy_test_acc_all[str(testing_noise_list[i])])))

    row += 2
    col = 0

    sheet.write(row, col, file_name + '_top5')
    for i in range (len(noisy_test_acc_5)):
        sheet.write(row, col+i+1, noisy_test_acc_5[i])

    row += 1

    for i in range (len(testing_noise_list)):
        col += 1
        for j in range(len(noisy_test_acc_5_all[str(testing_noise_list[i])])):
            sheet.write(row+j+1, col, noisy_test_acc_5_all[str(testing_noise_list[i])][j])

    row += (len(noisy_test_acc_5_all['0.02'])+1)

    col = 0
    for i in range(len(testing_noise_list)):
        sheet.write(row, col+i+1, np.std(np.asarray(noisy_test_acc_5_all[str(testing_noise_list[i])])))

    book.save('./excel/'+ file_name + '.xls')
