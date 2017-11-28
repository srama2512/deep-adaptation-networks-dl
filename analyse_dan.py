import os
import pdb
import json
import math
import torch
import argparse
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from misc.model import *
from misc.utils import *
from tensorboardX import SummaryWriter

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def evaluate(opts, print_result=True):
    if opts.dataset == 'cifar' or opts.dataset == 'svhn':
        opts.num_classes = 10
    elif opts.dataset == 'sketches':
        opts.num_classes = 250
    elif opts.dataset == 'caltech':
        opts.num_classes = 257
    
    temp_opts = argparse.Namespace()
    temp_opts.num_classes = 10 # Just a dummy
    temp_opts.init = 'xavier' # Just a dummy
    temp_opts.pretrained = True # Just a dummy
    temp_opts.num_fc = opts.num_fc
    base_net = VGG_B(temp_opts)
    
    opts.base_net = base_net
    opts.init = 'xavier' # Just a dummy

    net = DAN_Model(opts)
    
    if not opts.load_model == '':
        chkpt = torch.load(opts.load_model)
        net.load_state_dict(chkpt)
    else:
        raise ValueError('Enter a valid model path!')

    if opts.base_network == 'imagenet':
        for i in range(len(net.features)):
            if isinstance(net.features[i], DAN_Module):
                const_weight = net.features[i].constant_weight_buffer.view(net.features[i].filter_shape)
                base_weight = base_net.features[i].weight.data
                assert((const_weight - base_weight).sum() == 0)
        print('Successfully checked!')
    elif opts.base_network == 'noise':
        for i in range(len(net.features)):
            if isinstance(net.features[i], DAN_Module):
                const_weight = net.features[i].constant_weight_buffer.view(net.features[i].filter_shape)
                fin, fout = nn.init._calculate_fan_in_and_fan_out(const_weight)
                true_std = math.sqrt(2.0/(fin+fout))
                curr_std = const_weight.std()
                assert abs(true_std - curr_std) < true_std/20
        print('Successfully checked!')
    return net, base_net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', type=str, default='xavier')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--dataset_path', type=str, default='/work/05147/srama/shareddir/data/cifar-10')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--num_fc', type=int, default=3)
    parser.add_argument('--base_network', type=str, default='imagenet')
    parser.add_argument('--base_network_path', type=str, default='')

    opts = parser.parse_args()
    opts.mean = [0.485, 0.456, 0.406]
    opts.std = [0.229, 0.224, 0.225]

    net, base_net = evaluate(opts)
