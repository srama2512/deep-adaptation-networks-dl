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
	optimizer = optim.Adam(net.parameters(), lr=lr)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
	return criterion, optimizer, exp_lr_scheduler

def svhn_label_transform(x):
    if x[0] == 10:
        x[0] = 0
    return x

def get_dataloaders(opts):
    if opts.dataset == 'cifar':
        transform = transforms.Compose(
                        [transforms.Scale((64, 64)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean = opts.mean, std = opts.std)])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
        return trainloader, testloader

    elif opts.dataset == 'svhn':
        train_transform = transforms.Compose(
                          [transforms.Scale((80, 80)),
                           transforms.RandomSizedCrop(64),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(), 
                           transforms.Normalize(mean = opts.mean, std = opts.std)])

        valid_transform = transforms.Compose(
						  [transforms.Scale((80, 80)),
                           transforms.CenterCrop(64),
						   transforms.ToTensor(),
						   transforms.Normalize(mean = opts.mean, std = opts.std)])

        test_transform = transforms.Compose(
                        [transforms.Scale((80, 80)),
                         transforms.CenterCrop(64),
                         transforms.ToTensor(),
                         transforms.Normalize(mean = opts.mean, std = opts.std)])

        trainset = ImageFolder(root=os.path.join(opts.dataset_path,'train'), 
                               transform=train_transform)
        trainloader = DataLoader(trainset, batch_size=opts.batch_size,
                                 shuffle=True, num_workers=opts.num_workers)

        validset = ImageFolder(root=os.path.join(opts.dataset_path, 'val'),
                               transform=valid_transform)
        validloader = DataLoader(validset, batch_size=opts.batch_size,
                                 shuffle=False, num_workers=opts.num_workers)
        #testset = ImageFolder(root=os.path.join(opts.dataset_path, 'test'),
        #                      transform=test_transform)
        #testloader = DataLoader(testset, batch_size=opts.batch_size,
        #                        shuffle=False, num_workers=opts.num_workers)

        return trainloader, validloader#, testloader

    elif opts.dataset == 'sketch':
        transform = transforms.Compose(
                        [transforms.Scale((64, 64)),
                         transforms.ToTensor()])

        trainset = torchvision.datasets.ImageFolder(root='./data/sketch', transform=transform)
        trainloader = DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

        return trainloader, None
	
    elif opts.dataset == 'caltech':
        transform = transforms.Compose(
                        [transforms.Scale((64, 64)),
                         transforms.ToTensor()])

        trainset = torchvision.datasets.ImageFolder(root='./data/256_ObjectCategories', transform=transform)
        trainloader = DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

        return trainloader, None
    
    else:
        raise(NameError('Dataset %s does not exist!'%(opts.dataset)))

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

            # print statistics
            running_loss += loss.data[0]
            if (i+1) % 100 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        
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

    opts = parser.parse_args()
    opts.mean = [0.485, 0.456, 0.406]
    opts.std = [0.229, 0.224, 0.225]

    json.dump(vars(opts), open(os.path.join(opts.save_path, 'opts.json'), 'w'))
    train(opts)
