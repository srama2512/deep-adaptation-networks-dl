from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from argparse import Namespace
import torch.nn as nn
import model as Model
import torch
import copy

opts = Namespace()
opts.pretrained = True
opts.init = 'xavier'

for num_classes in [2, 10, 43, 250, 256, 1623]:
    opts.num_classes = num_classes

    vggb = Model.VGG_B(opts)

    vggb_params_count = 0

    for layer in vggb.parameters():
        vggb_params_count += layer.nelement()

    print('[%d classes] Number of params in VGGB: %d'%(num_classes, vggb_params_count))
