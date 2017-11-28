from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import pdb

class Net(nn.Module):
    def __init__(self, opts):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, opts.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def ixvr(input_layer, bias_val=0.1):
    if not str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm2d'>":
        if hasattr(input_layer, 'weight'):
            nn.init.xavier_normal(input_layer.weight);
        if hasattr(input_layer, 'bias'):
            nn.init.constant(input_layer.bias, bias_val);
    return input_layer

def inrml(input_layer, mean=0, std=0.01):
    if not str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm2d'>":
        if hasattr(input_layer, 'weight'):
            nn.init.normal(input_layer.weight, mean, std);
        if hasattr(input_layer, 'bias'):
            nn.init.constant(input_layer.bias, 0.01);
    return input_layer

class VGG_B(nn.Module):

    def __init__(self, opts):
        super(VGG_B, self).__init__()
        net = models.vgg13_bn(pretrained=opts.pretrained)
        if opts.pretrained:
            self.features = net.features
        else:
            if opts.init == 'xavier':
                self.features = nn.Sequential(*(ixvr(net.features[i]) \
                                                for i in range(len(net.features))))
            else:
                self.features = nn.Sequential(*(inrml(net.features[i]) \
                                                for i in range(len(net.features))))
         
        self.classifier = []
        #Add the FC layers
        if opts.init == 'xavier':
            for i in range(opts.num_fc-1):
                self.classifier.append(ixvr(nn.Linear(2048, 2048)))
                self.classifier.append(nn.ReLU(inplace=True))
                self.classifier.append(nn.Dropout(p=0.5))

            self.classifier.append(ixvr(nn.Linear(2048, opts.num_classes)))
        else:
            for i in range(opts.num_fc-1):
                self.classifier.append(inrml(nn.Linear(2048, 2048)))
                self.classifier.append(nn.ReLU(inplace=True))
                self.classifier.append(nn.Dropout(p=0.5))
           
            self.classifier.append(inrml(nn.Linear(2048, opts.num_classes)))
        
        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 2048)
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
        nout = w_size[0]
        self.weight = nn.Parameter(torch.randn(nout, nout)*0.01)
        self.bias = nn.Parameter(torch.randn(nout)*0.01)
        self.stride = nn_module.stride
        self.padding = nn_module.padding
        filter_shape = w_size
        self.filter_shape = [filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]]

        #self.constant_weight = Variable(params[0].data.view(self.filter_shape[0], -1), requires_grad=False)
        self.register_buffer('constant_weight_buffer', params[0].data.view(self.filter_shape[0], -1))
        self.constant_weight = Variable(self.constant_weight_buffer, requires_grad=False)

    # Redefine apply to modify constant_weight as well
    def _apply(self, fn):
        self = super(DAN_Module, self)._apply(fn)
        self.constant_weight = Variable(self.constant_weight_buffer, requires_grad=False)#fn(self.constant_weight)
        return self

    # Redefine load_state_dict to re-assign constant_weight as well
    def load_state_dict(self, state_dict, strict=True):
        super(DAN_Module, self).load_state_dict(state_dict, strict)
        self.constant_weight = Variable(self.constant_weight_buffer, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, weight=torch.matmul(self.weight, self.constant_weight).view(-1, *self.filter_shape[1:]), \
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
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.reset_parameters()
                self.features.append(layer)
            else:
                self.features.append(layer)
        
        self.features = nn.Sequential(*self.features)
        # Randomly initialize the fully connected layers
        if opts.init == 'xavier':
            self.classifier = [ixvr(copy.deepcopy(base_net.classifier[i])) \
                                              for i in range(len(base_net.classifier)-1)]
            self.classifier.append(ixvr(nn.Linear(base_net.classifier[-1].weight.size(1), opts.num_classes)))
        else:
            self.classifier = [inrml(copy.deepcopy(base_net.classifier[i])) \
                                              for i in range(len(base_net.classifier)-1)]
            self.classifier.append(inrml(nn.Linear(base_net.classifier[-1].weight.size(1), opts.num_classes)))      

        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.classifier[0].in_features)
        x = self.classifier(x)

        return x
