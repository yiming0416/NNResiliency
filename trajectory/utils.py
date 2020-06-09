import torch
import numpy as np
from networks import *

def extract_param_vec(net: torch.nn.Module):
    return torch.cat([p.detach().view(-1).float() for p in net.parameters()], dim=0)

def extract_grad_vec(net: torch.nn.Module):
    try:
        return torch.cat([p.grad.detach().view(-1).float() for p in net.parameters()], dim=0)
    except: # Basically it's because p.grad is None
        return torch.zeros(num_parameters(net), device=next(net.parameters()).device)

# TODO: write a more efficient MNIST DataLoader
def evaluate_with_lambda(net: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion):
    net.eval()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    avg_acc5 = AverageMeter()
    device = next(net.parameters()).device
    num_data = 0
    total_loss = 0
    K_mat = torch.zeros(num_parameters(net), num_parameters(net), device=device)
    for data, label in dataloader:
        data, label = data.to(device), label.to(device)
        output = net(data)
        loss = criterion(output, label)
        batch_size = data.size(0)
        num_data += batch_size
        total_loss = (total_loss + loss * batch_size) / num_data
    total_loss.backward()
    g = extract_grad_vec(net)
    for data, label in dataloader:
        net.zero_grad()
        data, label = data.to(device), label.to(device)
        output = net(data)
        loss = criterion(output, label)
        loss.backward()
        g_i = extract_grad_vec(net)
        d = (g_i - g).to(device)
        K_mat.addr_(d, d)
        # K_mat += d.ger(d) / len(dataloader)

        prec1, prec5 = accuracy(output.data, label.data, topk=(1,5))
        avg_loss.update(loss.item(), data.size(0))
        avg_acc.update(prec1.item())
        avg_acc5.update(prec5.item())
    s, v = torch.lobpcg(K_mat, k=1)
    return avg_loss.avg, avg_acc.avg, avg_acc5.avg, s.item()

def evaluate(net: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion):
    net.eval()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    avg_acc5 = AverageMeter()
    device = next(net.parameters()).device
    num_data = 0
    total_loss = 0
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = net(data)
            loss = criterion(output, label)

            prec1, prec5 = accuracy(output.data, label.data, topk=(1,5))
            avg_loss.update(loss.item(), data.size(0))
            avg_acc.update(prec1.item())
            avg_acc5.update(prec5.item())
    return avg_loss.avg, avg_acc.avg, avg_acc5.avg

def sample_near_trajectories(trajectories: torch.Tensor, num_samples: int, stdev: float=1) -> torch.Tensor:
    num_points, dim = trajectories.shape
    selected_point_inds = torch.randint(low=0, high=num_points, size=(num_samples,))
    samples = trajectories[selected_point_inds,:].add_(torch.randn(selected_point_inds.nelement(), dim).mul_(stdev))
    return samples

def split_rows_generator(ndarray, lengths):
    assert np.all(np.array(lengths) > 0)
    assert ndarray.shape[0] == sum(lengths)
    idx = 0
    for l in lengths:
        yield ndarray[idx:idx+l]
        idx += l