# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:30:05 2019

@author: intisar
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from expertnet import ExpertNet
from resnet import resnet20
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import  SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import time
import copy
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--subset', type=int, default=0, metavar='N',
                    help='Which subset to train/test on')


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

test_set = datasets.CIFAR10('./data', train=False, download=True,
                   transform=transform_test)



train_set = datasets.CIFAR10('./data', train=True, download=True,
                   transform=transform_train)

classes = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
           'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}




########################################
SUBSET = []
for _ in range(30):
    SUBSET.append([])
#SUBSET[0] = {5, 7, 3, 4}
#SUBSET[1] = {9, 1, 8, 6}
#SUBSET[2] = {0,2}

######### create the round robin subset #######
for i in range(10):
    elem = i
    for _ in range(2):
        SUBSET[i].append(elem)
        elem = (elem + 1)%10
for i in range(10, 20):
    elem = i-10
    for _ in range(3):
        SUBSET[i].append(elem)
        elem = (elem + 1)%10  

for i in range(20, 30):
    elem = i-20
    for _ in range(4):
        SUBSET[i].append(elem)
        elem = (elem + 1)%10  
        
for sub in SUBSET:
    print (sub)
    



def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in TestLoaderforExpert:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        output = F.softmax(output)
        test_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = torch.argsort(output, dim=1, descending=True)[0:, 0]
   
        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
   
    test_loss /= tot_test
    print('\nTest set: Average loss: {:.4f}, TOP 1 Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, tot_test,
         100. * correct.double() / tot_test ))
    return (correct.double() / tot_test)


### IMPORT THE TEACHER NETWORK FOR THE SOFT TARGET TRAINING
    
super_set = [0,1,2,3,4,5,6,7,8,9]
for sub in SUBSET:
    ## SELECT THE INDICES OF THE CLASS YOU WANT TO TRAIN THE EXPERT MODULE ON
    indices_train = [i for i,e in enumerate(train_set.targets) if e in sub] 
    tot_train = len(indices_train)
    indices_test = [j for j,k in enumerate(test_set.targets) if k in sub]
    tot_test = len(indices_test)
    
    
    TrainLoaderforExpert = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                             transform=transform_train),
                             batch_size=args.batch_size, sampler = SubsetRandomSampler(indices_train))
    TestLoaderforExpert = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transform_test),
            batch_size=args.test_batch_size, sampler = SubsetRandomSampler(indices_test))
    
    complement_set = np.setdiff1d(super_set, sub)
    print (complement_set, sub)
    print (sub)
    TrainLoaderforAll = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                             transform=transform_train),
                             batch_size=args.batch_size, sampler = SubsetRandomSampler(complement_set))
    
    ## a single instance of Expert Net for inference
    model = resnet20()
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=5e-4, nesterov=True)
    scheduler = StepLR(optimizer, step_size=35, gamma=0.1)

    from resnet import resnet56
    teacher = resnet56()
    teacher = teacher.cuda()
    teacher_weights = torch.load('./weights/suSan.pth.tar')
    teacher.load_state_dict(teacher_weights)
    
    args.epochs =  1
    val_scores = []
    wt = "./weights/rr/random_injection/rr_subset_" + str(sub) + ".pth.tar"
    #par = torch.load(wt)
    #model.load_state_dict(par)
    #print ("TEST SET performance on: {}".format(wt))
    ###### Start training the network, then validation and test #####
    best_correct = -999
    for epoch in range(args.epochs):
        test(model)
   

model_size = sum( p.numel() for p in model.parameters() if p.requires_grad)

print("The size of the expert model: {}M".format(model_size/1000000))
    


