import os
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def get_dataloaders(opts):
    if opts.dataset == 'svhn' or opts.dataset == 'cifar' or opts.dataset == 'sketch' or opts.dataset == 'caltech':
        train_transform = transforms.Compose(
                          [transforms.Scale((72, 72)),
                           transforms.RandomSizedCrop(64),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(), 
                           transforms.Normalize(mean = opts.mean, std = opts.std)])

        valid_transform = transforms.Compose(
						  [transforms.Scale((72, 72)),
                           transforms.CenterCrop(64),
						   transforms.ToTensor(),
						   transforms.Normalize(mean = opts.mean, std = opts.std)])

        test_transform = transforms.Compose(
                        [transforms.Scale((72, 72)),
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
    
    else:
        raise(NameError('Dataset %s does not exist!'%(opts.dataset)))

