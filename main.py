import argparse
import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from misc.model import *

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def loss_optim(net, lr, momentum):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=lr)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
	return criterion, optimizer, exp_lr_scheduler

def get_model(opts):
	net = Net(opts)
	return net

def svhn_label_transform(x):
	if x[0] == 10:
		x[0] = 0
	return x

def get_dataloaders(opts):
	if opts.dataset == 'cifar':
		transform = transforms.Compose(
						[transforms.Scale((64, 64)),
						 transforms.ToTensor(),
						 transforms.Normalize(mean = [0.53129727, 0.52593911, 0.52069134], std = [0.28938246, 0.28505746, 0.27971658])])

		trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
		                                download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
		                                  shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root='./data', train=False,
		                               download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=4,
		                                 shuffle=False, num_workers=2)
		return trainloader, testloader

	if opts.dataset == 'svhn':
		transform = transforms.Compose(
						[transforms.Scale((64, 64)),
						 transforms.ToTensor(),
						 transforms.Normalize(mean = [0.43768218, 0.44376934, 0.47280428], std = [0.1980301, 0.2010157, 0.19703591])])

		trainset = torchvision.datasets.SVHN(root='./data', split='train',
		                                download=True, transform=transform, target_transform=svhn_label_transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
		                                  shuffle=True, num_workers=2)

		testset = torchvision.datasets.SVHN(root='./data', split='test',
		                               download=True, transform=transform, target_transform=svhn_label_transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=4,
		                                 shuffle=False, num_workers=2)
		return trainloader, testloader
	if opts.dataset == 'sketch':
		transform = transforms.Compose(
						[transforms.Scale((64, 64)),
						 transforms.ToTensor()])

		trainset = torchvision.datasets.ImageFolder(root='./data/sketch', transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
		                                  shuffle=True, num_workers=2)

		return trainloader, None
	if opts.dataset == 'caltech':
		transform = transforms.Compose(
						[transforms.Scale((64, 64)),
						 transforms.ToTensor()])

		trainset = torchvision.datasets.ImageFolder(root='./data/256_ObjectCategories', transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
		                                  shuffle=True, num_workers=2)

		return trainloader, None


def evaluate(net, dataloader, cuda_flag=False, print_result=True):
	correct = 0
	total = 0
	net.eval()
	for data in dataloader:
		images, labels = data
		if cuda_flag:
			images = images.cuda()
			labels = labels.cuda()
		outputs = net(Variable(images))
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
	net = get_model(opts)
	trainloader, testloader = get_dataloaders(opts)

	criterion, optimizer, exp_lr_scheduler = loss_optim(net, opts.lr, opts.momentum)

	for epoch in range(opts.epochs):  # loop over the dataset multiple times
		running_loss = 0.0
		exp_lr_scheduler.step()
		for i, data in enumerate(trainloader, 0):
			# get the inputs
			inputs, labels = data

			# wrap them in Variable
			inputs, labels = Variable(inputs), Variable(labels)
			if(opts.dataset == 'svhn'):
				labels = labels.type(torch.LongTensor)
				labels = labels.view(-1)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.data[0]
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

		evaluate(net, trainloader)
		if testloader != None:
			evaluate(net, testloader)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--pretrained', type=bool, default=False)
	parser.add_argument('--init', type=str, default='xavier')
	parser.add_argument('--dataset', type=str, default='cifar')
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--momentum', type=float, default=0.9)

	opts = parser.parse_args()
	train(opts)