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
opts.num_classes = 256
vggb = Model.VGG_B(opts)

caltech_count = 0
vggb_params_count = 0
for layer in vggb.parameters():
    caltech_count += layer.nelement()

total_count = 0
for num_classes in [2, 10, 10, 43, 121, 250, 1623]:

    opts.num_classes = num_classes

    vggb_params_count = 0
    for layer in vggb.parameters():
        vggb_params_count += layer.nelement()
    
    opts.base_net = vggb
    dan_vggb = Model.DAN_Model(opts)

    dan_params_count = 0
    for layer in dan_vggb.parameters():
        dan_params_count += layer.nelement()

    print('[%4d classes] # Parameters - VGGB: %d , DAN: %d (%.2f %%)'%(num_classes, vggb_params_count, dan_params_count, 100.0*float(dan_params_count)/vggb_params_count))
    total_count += dan_params_count

print('Relative gain: %.2f %%'%(float(total_count) / caltech_count)) 

