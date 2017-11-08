import torch
import model as Models
from argparse import Namespace
from torch.autograd import Variable

# Check loading of pre-trained network
opts = Namespace()
opts.pretrained = True
opts.num_classes = 100
opts.init = 'xavier'

net = Models.VGG_B(opts)

# Check forward pass
inputV = Variable(torch.randn(10, 3, 224, 224))
outputV = net(inputV)
"""
# Check loading of randomly initialized network
opts = Namespace()
opts.pretrained = False
opts.num_classes = 100
opts.init = 'xavier'

net = Models.VGG_B(opts)

# Check forward pass
inputV = Variable(torch.randn(10, 3, 224, 224))
outputV = net(inputV)
"""

# Check if DAN module works
conv2d_dummy = torch.nn.Conv2d(3, 32, 5, 1, 2)
dan_dummy = Models.DAN_Module(conv2d_dummy)

# Check forward pass works of DAN module
x_in = Variable(torch.randn(10, 3, 224, 224))
x_out = dan_dummy(x_in)
assert(x_out.size() == torch.Size([10, 32, 224, 224]))

# Check creation of DAN
opts = Namespace()
opts.base_net = net
opts.cuda = False
opts.init = 'xavier'
net_DAN = Models.DAN_Model(opts)

# Check forward pass of DAN
x_in = Variable(torch.randn(10, 3, 224, 224))
x_out = net_DAN(x_in)
print('DAN Output size:', x_out.size())
