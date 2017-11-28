from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from argparse import Namespace
import torch.nn as nn
import model as Model
import torch
import copy
import pdb

opts = Namespace()
opts.pretrained = True
opts.init = 'xavier'
base_dataset = 'Imagenet'
opts.num_classes = 1000
vggb = Model.VGG_B(opts)

caltech_count = 0
vggb_params_count = 0
for layer in vggb.parameters():
    caltech_count += layer.nelement()

total_count = 0
dataset_to_classes = {"Cifar-10": 10, "GTSR": 43, "SVHN": 10, "Caltech-256": 257, "Dped": 2, "Omniglot": 1623, "Sketches": 250, 'Plankton': 103}

for dataset, num_classes in dataset_to_classes.iteritems():
    if dataset != base_dataset:
        opts.num_classes = num_classes

        vggb_params_count = 0
        for layer in vggb.parameters():
            vggb_params_count += layer.nelement()
        
        opts.base_net = vggb
        dan_vggb = Model.DAN_Model(opts)

        dan_params_count = 0
        for layer in dan_vggb.parameters():
            dan_params_count += layer.nelement()

        print('[Dataset: %s, %4d classes] # Parameters - VGGB: %d , DAN: %d (%.2f %%)'%(dataset, num_classes, vggb_params_count, dan_params_count, 100.0*float(dan_params_count)/vggb_params_count))
        total_count += dan_params_count

print('# param: %.2f'%(1 + float(total_count) / caltech_count)) 

