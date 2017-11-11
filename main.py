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

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def loss_optim(net, lr, momentum):
    criterion = nn.CrossEntropyLoss()
    list_of_dicts = []
    list_of_dicts.append({'params': net.features.parameters(), 'lr': lr})
    list_of_dicts.append({'params': net.classifier.parameters(), 'lr': 0.1*lr})

    optimizer = optim.Adam(list_of_dicts)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return criterion, optimizer, exp_lr_scheduler

def svhn_label_transform(x):
    if x[0] == 10:
        x[0] = 0
    return x

def evaluate(net, dataloader, opts, print_result=True):
    correct = 0
    total = 0 
    net.eval()
    for data in dataloader:
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
        print('==> Accuracy: %f %% ' % (100 * float(correct) / float(total)))
    net.train()
    return accuracy

def train(opts):
    if opts.dataset == 'cifar' or opts.dataset == 'svhn':
        opts.num_classes = 10
    elif opts.dataset == 'sketch':
        opts.num_classes = 250
    elif opts.dataset == 'caltech':
        opts.num_classes = 257
    net = VGG_B(opts)
    if not opts.load_model == '':
        chkpt = torch.load(opts.load_model)
        net.load_state_dict(chkpt)

    trainloader, validloader = get_dataloaders(opts)

    criterion, optimizer, exp_lr_scheduler = loss_optim(net, opts.lr, opts.momentum)

    net.train()
    best_valid_accuracy = 0

    writer = SummaryWriter(log_dir = opts.save_path)

    dummy_input = Variable(torch.randn(1, 3, 64, 64))
    if opts.cuda:
        net = net.cuda()
        criterion = criterion.cuda()
        dummy_input = dummy_input.cuda()

    dummy_output = net(dummy_input)
    writer.add_graph(net, dummy_output)
    
    iters = 0
    for epoch in range(opts.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        exp_lr_scheduler.step()
            
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # convert to cuda if available
            if opts.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            #if(opts.dataset == 'svhn'):
            #    labels = labels.type(torch.LongTensor)
            #    labels = labels.view(-1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            grad_of_params = {}
            for name, parameter in net.named_parameters():
                grad_of_params[name] = parameter.grad
            
            # print statistics
            running_loss += loss.data[0]
            if (i+1) % 50 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 50))

                writer.add_scalar('data/train_loss', running_loss/50, iters)
                running_loss = 0.0

            iters += 1
            for name, param in net.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), iters)

        train_accuracy = evaluate(net, trainloader, opts, False)
        valid_accuracy = evaluate(net, validloader, opts, False)

        print('===> Epoch: %3d,     Train Accuracy: %.4f,     Valid Accuracy: %.4f'%(epoch, train_accuracy, valid_accuracy))

        writer.add_scalar('data/train_accuracy', train_accuracy, epoch)
        writer.add_scalar('data/valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('data/train_loss', running_loss, epoch)

        if best_valid_accuracy <= valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(net.state_dict(), os.path.join(opts.save_path, 'model_best.net'))

        torch.save(net.state_dict(), os.path.join(opts.save_path, 'model_latest.net'))

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', type=str, default='xavier')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--dataset_path', type=str, default='/work/05147/srama/shareddir/data/svhn')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pretrained', type=str2bool, default=True, help='use imagenet pretrained weights?')
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--load_model', type=str, default='')

    opts = parser.parse_args()
    opts.mean = [0.485, 0.456, 0.406]
    opts.std = [0.229, 0.224, 0.225]

    json.dump(vars(opts), open(os.path.join(opts.save_path, 'opts.json'), 'w'))
    train(opts)
