import os
import pdb
import json
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

def svhn_label_transform(x):
    if x[0] == 10:
        x[0] = 0
    return x

def evaluate(opts, print_result=True):
    if opts.dataset == 'cifar' or opts.dataset == 'svhn':
        opts.num_classes = 10
    elif opts.dataset == 'sketches':
        opts.num_classes = 250
    elif opts.dataset == 'caltech':
        opts.num_classes = 257
    net = VGG_B(opts)
    if not opts.load_model == '':
        chkpt = torch.load(opts.load_model)
        net.load_state_dict(chkpt)
    
    trainloader, validloader, testloader = get_dataloaders(opts)
    
    if opts.cuda:
        net = net.cuda()

    correct = 0
    total = 0 
    net.eval()
    for data in testloader:
        images, labels = data
        if opts.cuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    
    accuracy = 100 * float(correct) / float(total)
    if print_result:
        print('==> Accuracy: %f %% ' % (accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', type=str, default='xavier')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--dataset_path', type=str, default='/work/05147/srama/shareddir/data/cifar-10')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pretrained', type=str2bool, default=True, help='use imagenet pretrained weights?')
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--num_fc', type=int, default=3)

    opts = parser.parse_args()
    opts.mean = [0.485, 0.456, 0.406]
    opts.std = [0.229, 0.224, 0.225]

    evaluate(opts)
