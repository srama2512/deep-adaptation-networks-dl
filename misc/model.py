from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy

def ixvr(input_layer, bias_val=0):
    if hasattr(input_layer, 'weight'):
        nn.init.xavier_normal(input_layer.weight);
    if hasattr(input_layer, 'bias'):
        nn.init.constant(input_layer.bias, bias_val);
    return input_layer

def inrml(input_layer, mean=0, std=0.005):
    if hasattr(input_layer, 'weight'):
        nn.init.normal(input_layer.weight, mean, std);
    if hasattr(input_layer, 'bias'):
        nn.init.constant(input_layer.bias, 1);
    return input_layer

class VGG_B(nn.Module):

    def __init__(self, opts):
        super(VGG_B, self).__init__()
        net = models.vgg13(pretrained=opts.pretrained)

        if opts.pretrained:
            self.features = net.features
            # Do not copy last layer for classification
            self.classifier = [net.classifier[i] \
                                              for i in range(len(net.classifier)-1)]
        else:
            if opts.init == 'xavier':
                self.features = nn.Sequential(*(ixvr(net.features[i]) \
                                                for i in range(len(net.features))))
                self.classifier = [ixvr(net.classifier[i]) \
                                                  for i in range(len(net.classifier)-1)]
            else:
                self.features = nn.Sequential(*(inrml(net.features[i]) \
                                                for i in range(len(net.features))))
                self.classifier = [inrml(net.classifier[i]) \
                                                  for i in range(len(net.classifier)-1)]

        # Add the final classification layer
        if opts.init == 'xavier':
            self.classifier.append(ixvr(nn.Linear(4096, opts.num_classes)))
        else:
            self.classifier.append(inrml(nn.Linear(4096, opts.num_classes)))
        
        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 25088)
        x = self.classifier(x)
        return x

class DAN_Module(nn.Module):
    """
    DAN Module on top of a Conv2d module. It takes in a Conv2d
    module as input and creates a new module that uses convolution
    weights as a linear combination of the weights from the Conv2d
    module. The biases are initialized randomly.
    """
    def __init__(self, nn_module):
        super(DAN_Module, self).__init__()
        params = list(nn_module.parameters())
        w_size = params[0].size()
        # Num input channels * filter width * filter height
        nout = w_size[1]*w_size[2]*w_size[3]
        self.weight = nn.Parameter(torch.randn(nout, nout)*0.01)
        self.bias = nn.Parameter(torch.randn(w_size[0])*0.01)
        self.stride = nn_module.stride
        self.padding = nn_module.padding
        filter_shape = w_size
        self.filter_shape = [filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]]

        self.constant_weight = Variable(params[0].data.view(self.filter_shape[0], -1), requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, weight=torch.matmul(self.constant_weight, self.weight).view(-1, *self.filter_shape[1:]), \
                             bias=self.bias, stride=self.stride, padding=self.padding)
        return x

class DAN_Model(nn.Module):

    def __init__(self, opts):
        super(DAN_Model, self).__init__()
        # The base network has to be provided as an input
        base_net = opts.base_net
        self.features = []
        # Assuming that the base_net has features and classifier separately
        for layer in base_net.features:
            if isinstance(layer, torch.nn.Conv2d):
                self.features.append(DAN_Module(layer))
            else:
                self.features.append(layer)
        
        self.features = nn.Sequential(*self.features)
        # Randomly initialize the fully connected layers
        if opts.init == 'xavier':
            self.classifier = nn.Sequential(*(ixvr(copy.deepcopy(base_net.classifier[i])) \
                                              for i in range(len(base_net.classifier))))
        else:
            self.classifier = nn.Sequential(*(inrml(copy.deepcopy(base_net.classifier[i])) \
                                              for i in range(len(base_net.classifier))))
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.classifier[0].in_features)
        x = self.classifier(x)

        return x
