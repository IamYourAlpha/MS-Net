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
from resnet import resnet20, resnet44
import torch.optim as optim
from torchvision import datasets
import transforms
from torch.utils.data import  SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import time
import random
import copy
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
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

# Random Erasing
parser.add_argument('--p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)




kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(probability = args.p, sh = args.sh, r1 = args.r1, ),
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


#transform_train = transforms.Compose([
#    transforms.RandomCrop(32,4),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225]),
#])
#
#
#transform_test = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225]),
#])

test_set = datasets.FashionMNIST('./data', train=False, download=True,
                   transform=transform_test)



train_set = datasets.FashionMNIST('./data', train=True, download=True,
                   transform=transform_train)

classes = {'T-shirt': 0, 'Trouser': 1, 'Pullover': 2, 'Dress': 3,
           'Coat': 4, 'Sandal': 5, 'Shirt': 6, 'Sneaker': 7, 'Bag': 8, 'Ankle boot': 9}




########################################
SUBSET = []
for _ in range(30):
    SUBSET.append([])



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



def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1)) * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)



def train(epoch, model, loss_fn):
    model.train()
    # teacher.eval()
    bernouli = []
    train_loss = 0
    for i in range(50):
        bernouli.append(1)
    for i in range(50, 100):
        bernouli.append(0)

    for batch_idx, (data, target) in enumerate(TrainLoaderforExpert):
        data_all, target_all = next(iter(TrainLoaderforAll))
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            data_all, target_all = data_all.cuda(), target_all.cuda()

        data, target = Variable(data), Variable(target)
        data_all, target_all = Variable(data_all), Variable(target_all)

        optimizer.zero_grad()
        output_ex = model(data)
        output_all = model(data_all)
        # teacher_output = teacher(data_all)
        # teacher_output = teacher_output.detach()
        alpha = random.choice(bernouli)
        #loss_kd = loss_fn(output_all, target_all, teacher_output, T=5.0, alpha=0.2)
        loss_ce = F.cross_entropy(output_ex, target)
        loss_kd = F.cross_entropy(output_all, target_all)
        tot_loss = alpha*(loss_ce) + (1-alpha) * loss_kd
        train_loss += tot_loss.item()
        tot_loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(TrainLoaderforExpert.dataset),
                 100. * batch_idx / len(TrainLoaderforExpert), tot_loss.item()))
    return round(train_loss/tot_train, 2)


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
        print ("\n*********** NEW WEIGHTS SAVED ****************\n")
    return now_correct

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    now_correct = 0.0
    global best_correct

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
    now_correct = (correct.double()/tot_test)
    return now_correct
#    if best_correct < now_correct:
#        best_correct = now_correct
#        print (best_correct)
#        best_model_wts = copy.deepcopy(model.state_dict())
#        torch.save(best_model_wts, wt)



### IMPORT THE TEACHER NETWORK FOR THE SOFT TARGET TRAINING

super_set = [0,1,2,3,4,5,6,7,8,9]
for sub in SUBSET:
    ## SELECT THE INDICES OF THE CLASS YOU WANT TO TRAIN THE EXPERT MODULE ON
    indices_train = [i for i,e in enumerate(train_set.targets) if e in sub]
    tot_train = len(indices_train)
    indices_test = [j for j,k in enumerate(test_set.targets) if k in sub]
    tot_test = len(indices_test)


    TrainLoaderforExpert = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=True, download=True,
                             transform=transform_train),
                             batch_size=args.batch_size, sampler = SubsetRandomSampler(indices_train))
    TestLoaderforExpert = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transform_test),
            batch_size=args.test_batch_size, sampler = SubsetRandomSampler(indices_test))

    complement_set = np.setdiff1d(super_set, sub)
    print (complement_set, sub)
    trains = datasets.FashionMNIST('./data', train=False, transform=transform_test)
    TrainLoaderforAll = torch.utils.data.DataLoader(trains, batch_size=args.batch_size,
                                                    shuffle=True)
#    TrainLoaderforAll = torch.utils.data.DataLoader(args.train_batch
#            datasets.CIFAR10('./data', train=True, download=True,
#                             transform=transform_train),
#                             batch_size=args.batch_size, sampler = SubsetRandomSampler(complement_set))
#
    ## a single instance of Expert Net for inference
    model = resnet20()
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=60, gamma=0.1)

    # from resnet import resnet56
    # teacher = resnet56()
    # teacher = teacher.cuda()
    # teacher_weights = torch.load('./weights/suSan.pth.tar')
    # teacher.load_state_dict(teacher_weights)

    args.epochs =  180
    val_scores = []
    test_scores = []
    wt = "./weights/rr/random_injection_erasing/res20_fmnist_0.5/rr_subset_" + str(sub) + ".pth.tar"
    #par = torch.load(wt)
    #model.load_state_dict(par)
    #print ("TEST SET performance on: {}".format(wt))
    ###### Start training the network, then validation and test #####
    best_correct = -999
    print ("Training starting for class {}".format(sub))

    for epoch in range(args.epochs):
        score = train(epoch, model, loss_fn=distillation)
        scheduler.step()
        print('Current Epoch:', epoch, 'LR:', scheduler.get_lr())
        score_train = train_evaluate(model)
        score_train = score_train.item() # Just take the number.
        val_scores.append(score_train)
        start_time = time.time()
        score_test = test(model)
        score_test = score_test.item()
        test_scores.append(score_test)
    ### save the records ####
    fl = "./textfiles/rr/random_injection_erasing/res20_0.5/subset_" + str(sub) + "_train_result.txt"
    with open(fl, 'w') as f:
        for item in val_scores:
            f.write("%s\n" % item)

    fl = "./textfiles/rr/random_injection_erasing/res20_0.5/subset_" + str(sub) + "_test_result.txt"
    with open(fl, 'w') as f:
        for item in test_scores:
            f.write("%s\n" % item)


model_size = sum( p.numel() for p in model.parameters() if p.requires_grad)

print("The size of the expert model: {}M".format(model_size/1000000))
