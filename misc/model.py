from torch.autograd import Variable
import torchvison.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch

def ixvr(input_layer, bias_val=0):
    nn.init.xavier_normal(input_layer.weight);
    nn.init.constant(input_layer.bias, bias_val);
    return input_layer

def inrml(input_layer, mean=0, std=0.005):
    nn.init.normal(input_layer.weight, mean, std);
    nn.init.constant(input_layer.bias, 1);
    return input_layer

class VGG_B(nn.Module):

    def __init__(self, opts):
        super(VGG_B, self).__init__()
        net = models.vgg13(pretrained=opts.pretrained)

        if opts.pretrained:
            self.features = net.features
            # Do not copy last layer for classification
            self.classifier = nn.Sequential(*(net.classifier[i] \
                                              for i in range(len(net.classifier)-1)))
        else:
            if opts.init == 'xavier':
                self.features = nn.Sequential(*(ixvr(net.features[i]) \
                                                for i in range(len(net.features))))
                self.classifier = nn.Sequential(*(ixvr(net.classifier[i]) \
                                                  for i in range(len(net.classifier)-1)))
            else:
                self.features = nn.Sequential(*(inrml(net.features[i]) \
                                                for i in range(len(net.features))))
                self.classifier = nn.Sequential(*(inrml(net.classifier[i]) \
                                                  for i in range(len(net.classifier)-1)))

        # Add the final classification layer
        if opts.init == 'xavier':
            self.classifier.add(ixvr(nn.Linear(4096, opts.num_classes)))
        else:
            self.classifier.add(inrml(nn.Linear(4096, opts.num_classes)))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 25088)
        x = self.classifier(x)
        return x

class DAN_Model(nn.Module):

    def __init__(self, opts):
        super(DANModel, self).__init__()
        # The base network has to be provided as an input
        self.base_net = opts.base_net
        self.features = []
        # Assuming that the base_net has features and classifier separately
        for layer in self.base_net.features.modules():
            if isinstance(layer, torch.nn.Conv2d):
                params = list(layer.parameters())
                w_size = params[0].size()
                nout = w_size[1]*w_size[2]*w_size[3]
                # The weights are for the linear combination
                # Randomly initialize the biases
                if opts.cuda:
                    W = Variable(torch.randn(nout, nout).cuda()*0.01, requires_grad=True)
                    b = Variable(torch.randn(w_size[0]).cuda()*0.01, requires_grad=True)
                else:
                    W = Variable(torch.randn(nout, nout)*0.01, requires_grad=True)
                    b = Variable(torch.randn(w_size[0])*0.01, requires_grad=True)
                
                filter_shape = w_size
                filter_shape = [filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]]
                self.features.append({'coeff-weight': W, 'bias': b, 'weight': params.weight.view(filter_shape[0], -1), 'trainable': True, 'shape': filter_shape, 'stride': layer.stride, 'padding': layer.padding})
            else:
                self.features.append({'layer': layer, 'trainable': False})

        # Randomly initialize the fully connected layers
        if opts.init == 'xavier':
            self.classifier = nn.Sequential(*(ixvr(self.base_net.classifier[i]) \
                                              for i in range(len(self.base_net.classifier))))
        else:
            self.classifier = nn.Sequential(*(inrml(self.base_net.classifier[i]) \
                                              for i in range(len(self.base_net.classifier))))

    def forward(self, x):
        for layer in self.features:
            if layer['trainable']:
                x = F.conv2d(x, weight=torch.matmul(layer['weight'], layer['coeff-weight']).view(layer['shape']), \
                             bias=layer['bias'], stride=layer['stride'], padding=layer['padding'])
            else:
                x = layer(x)

        x = x.view(-1, self.classifier[0].in_features)
        x = self.classifier(x)

        return x
