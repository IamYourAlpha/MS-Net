# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:07:51 2019

@author: Intisar

This is a program to segregate the dataset into confusable class.


"""

from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import  SubsetRandomSampler

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)




kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



transform_train = transforms.Compose([
    transforms.RandomCrop(32,4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])

    
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])

classes = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
           'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

# SUBSET 1 = 5, 7, 3, 4
# SUBSET 2 = 9, 1, 8, 6
# SUBSET 3 = 0,2

test_set = datasets.CIFAR10('./data', train=False, download=True,
                   transform=transform_train)



train_set = datasets.CIFAR10('./data', train=False, download=True,
                   transform=transform_train)



indices = [i for i,e in enumerate(test_set.targets) if (e == 5  or e == 7 or e == 3 or e == 4)]


train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True,
                   transform=transform_train),
    batch_size=args.batch_size, sampler = SubsetRandomSampler(indices))

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transform_train),
    batch_size=args.test_batch_size, sampler = SubsetRandomSampler(indices))

for data, target in train_loader:
    print (target)
    break
  
    




