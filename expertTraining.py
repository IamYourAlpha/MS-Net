

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:07:51 2019

@author: Intisar

This is a program to segregate the dataset into confusable class.


"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet20, resnet56
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import  SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import time
import copy

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
parser.add_argument('--subset', type=int, default=7, metavar='N',
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

classes = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
           'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}


########################################
SUBSET = []
for _ in range(8):
    SUBSET.append([])
print (len(SUBSET))
SUBSET[0] = {5, 7, 3, 4}
SUBSET[1] = {9, 1, 8, 6}
SUBSET[2] = {0,2}
SUBSET[3] = {5, 7}
SUBSET[4] = {7, 4}
SUBSET[5] = {5, 4}
SUBSET[6] = {9, 1}
SUBSET[7] = {9, 8}

SUB = SUBSET[args.subset]

wt = "./weights/expert_resnet20_subset_" + str(args.subset) + ".pth.tar"
    
#########################################
test_set = datasets.CIFAR10('./data', train=False, download=True,
                   transform=transform_test)



train_set = datasets.CIFAR10('./data', train=True, download=True,
                   transform=transform_train)


## SELECT THE INDICES OF THE CLASS YOU WANT TO TRAIN THE EXPERT MODULE ON
indices_train = [i for i,e in enumerate(train_set.targets) if e in SUB] 
tot_train = len(indices_train)
indices_test = [j for j,k in enumerate(test_set.targets) if k in SUB]
tot_test = len(indices_test)
############## Dataset to train the expert ##########################################


TrainLoaderforExpert = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True,
                   transform=transform_train),
    batch_size=args.batch_size, sampler = SubsetRandomSampler(indices_train))

TestLoaderforExpert = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transform_test),
    batch_size=args.test_batch_size, sampler = SubsetRandomSampler(indices_test))

##################################################################################

# Import the model and load weights
model = resnet20()
if torch.cuda.is_available():
    model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=5e-4, nesterov=True)
scheduler = StepLR(optimizer, step_size=35, gamma=0.1)
#ck = torch.load(wt)
#ck = torch.load('./weights/suSan.pth.tar')
#model.load_state_dict(ck)



def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1)) * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)



def train(epoch, model):
    model.train()
    for batch_idx, (data, target) in enumerate(TrainLoaderforExpert):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.softmax(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(TrainLoaderforExpert.dataset),
                 100. * batch_idx / len(TrainLoaderforExpert), loss.item()))

best_correct = -999

def train_evaluate(model):
    model.eval()
    train_loss = 0
    correct = 0
    now_correct = 0
    global best_correct
    for data, target in TrainLoaderforExpert:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        train_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        train_loss, correct, tot_train,
        100. * correct.double() / tot_train))
    now_correct = correct.double()/tot_train
    if best_correct < now_correct:
        best_correct = now_correct
        print (best_correct)
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, wt)
    return now_correct

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
        #pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        pred = torch.argsort(output, dim=1, descending=True)[0:, 0]
        #pred2 = torch.argsort(output, dim=1, descending=True)[0:, 1] # top 2
        #pred3 = torch.argsort(output, dim=1, descending=True)[0:, 2] # top 3
        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        #correct += pred2.eq(target.data.view_as(pred2)).cpu().sum()
        #correct += pred3.eq(target.data.view_as(pred3)).cpu().sum()

    test_loss /= tot_test
    print('\nTest set: Average loss: {:.4f}, TOP 1 Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, tot_test,
         100. * correct.double() / tot_test ))
    return (correct.double() / tot_test)
#    
#    report = classification_report(y_pred, y_truth)
#    print (confusion_matrix(y_pred, y_truth))
#    print (report)



args.epochs = 100
val_scores = []
for epoch in range(1, args.epochs + 1):
     train(epoch, model)
     scheduler.step()
     print('Epoch:', epoch,'LR:', scheduler.get_lr())
     score = train_evaluate(model)
     score = score.item() # Just take the number.
     val_scores.append(score)
     start_time = time.time()
     test(model)

fl = "./textfiles/subset" + str(args.subset) + "_result.txt"
with open(fl, 'w') as f:
    for item in val_scores:
        f.write("%s\n" % item)


model_size = sum( p.numel() for p in model.parameters() if p.requires_grad)

print("The size of the Teacher Model: {}M".format(model_size))


#print("--- %s seconds ---" % (time.time() - start_time))

    








