#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import os
import sys
import math

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)

opts = parser.parse_args()

if opts.dataset == 'cifar':
	transform = transforms.Compose(
						[transforms.Scale((64, 64)),
						 transforms.ToTensor()])
	data = dset.CIFAR10(root='./data', train=True, download=True,
	                    transform=transform).train_data
	data = data.astype(np.float32)/255.

	means = []
	stdevs = []
	for i in range(3):
	    pixels = data[:,i,:,:].ravel()
	    means.append(np.mean(pixels))
	    stdevs.append(np.std(pixels))

	print("means: {}".format(means))
	print("stdevs: {}".format(stdevs))
	print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
elif opts.dataset == 'svhn':
	transform = transforms.Compose(
						[transforms.Scale((64, 64)),
						 transforms.ToTensor()])
	data = dset.SVHN(root='./data', split='train',
		                                download=True, transform=transform).data
	data = data.astype(np.float32)/255.

	means = []
	stdevs = []
	for i in range(3):
	    pixels = data[:,i,:,:].ravel()
	    means.append(np.mean(pixels))
	    stdevs.append(np.std(pixels))

	print("means: {}".format(means))
	print("stdevs: {}".format(stdevs))
	print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))