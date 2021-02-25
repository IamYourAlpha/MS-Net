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
from resnet import resnet20, resnet56, resnet44
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
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
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
parser.add_argument('--topn', type=int, default=3, metavar='N',
                    help='how many top n you want to further evaluate')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}





transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

test_set = datasets.FashionMNIST('./data', train=False, download=True,
                   transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=False, transform=transform_test),
    batch_size=args.test_batch_size, shuffle=True)
indices_test = [j for j,k in enumerate(test_set.targets) if k in [1,2]]

#test_loader = torch.utils.data.DataLoader(
#            datasets.CIFAR10('./data', train=False, transform=transform_test),
#            batch_size=args.test_batch_size, sampler = SubsetRandomSampler(indices_test))

classes = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
           'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}




########################################
#SUBSET = []
#for _ in range(10):
#    SUBSET.append([])

SUBSET = []
for _ in range(10):
    SUBSET.append([])

#SUBSET[0] =[5, 7, 3, 4]
#SUBSET[1] = [9, 1, 8, 6]
#SUBSET[2] = [0,2]
#SUBSET[3] = [5, 7]
#SUBSET[4] = [7, 4]
#SUBSET[5] = [5, 4]
#SUBSET[6] = [9, 1]
#SUBSET[7] = [9, 8]


######### create the round robin subset #######
# for i in range(10):
#     elem = i
#    for _ in range(2):
#        SUBSET[i].append(elem)
#        elem = (elem + 1)%10
#for i in range(10, 20):
#    elem = i-10
#    for _ in range(3):
#        SUBSET[i].append(elem)
       # elem = (elem + 1)%10
#
#for i in range(20, 30):
#    elem = i-20
#    for _ in range(4):
#        SUBSET[i].append(elem)
#        elem = (elem + 1)%10
for i in range(10):
    elem = i
    for _ in range(4):
        SUBSET[i].append(elem)
        elem = (elem + 1)%10





subset_flag = {}

for i in SUBSET:
    subset_flag[str(i)] = True




print (SUBSET)
import math

def trust_factor(sub_sz, mx_sz):

    if sub_sz < 3:
        return 1
    else:
        a = math.exp(-(sub_sz/mx_sz))
        return a


def test():
    router = resnet56()
    #rweights = torch.load('./weights/router_resnet20_all_class.pth.tar')
    #rweights = torch.load('./weights/suSan.pth.tar')
    start_time = time.time()
    #rweights = torch.load('./weights/best_so_far_res56.pth.tar')
    rweights = torch.load('./weights/resnet56_fmnist.pth.tar')
    router.load_state_dict(rweights)
    if torch.cuda.is_available():
        router.cuda()
    router.eval()
    test_loss = 0
    correct = 0
    tt = 0
    c = 0
    delta = []
    for data, target in (test_loader):
       # if c == 50:
       #     break
#        c = c + 1
        if (c % 20 == 0):
            print ("----- expert accuracy so far : {}/{}-----\n----- router accuracy so far : {}/{}-----"
                   .format(correct, c, tt, c))
            print ("The DELTA/improvement between router and expert: {}\n".format(abs(correct - tt)))
            if (c > 0):
                print ("Forcasting {:.2f}% (approx) accuracy at the end\n"
                       .format(((10000.00/c)*abs(tt- correct))/100.0 + 93.88))

        c = c + 1
        delta.append(abs(correct - tt))
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = router(data)
        output = F.softmax(output)
        #print (output) #################### Remove this while running
        rsoftmax = torch.sort(output, dim=1, descending=True)[0][0:, 0:args.topn]
        pred = output.data.max(1, keepdim=True)[1]
        pred2 = torch.argsort(output, dim=1, descending=True)[0][1]
        tt += pred.eq(target.data.view_as(pred)).cpu().sum()
        #tt += pred2.eq(target.data.view_as(pred)).cpu().sum()

        sortedsoftmax = torch.argsort(output, dim=1, descending=True)[0:1, 0:args.topn]
        sortedsoftmax = np.array(sortedsoftmax.cpu())

        ## reSet/call the predicitons
        predictions = []
        for i in SUBSET:
            subset_flag[str(i)] = True

        for i, pred in enumerate(sortedsoftmax):
            for j in range(args.topn):
                predictions.append(pred[j])
        #print ("The top {} predictions of router: {}".format(args.topn, predictions))
        rsm = []
        for i, pred in enumerate(rsoftmax):
            for j in range(args.topn):
                rsm.append(pred[j])


        #sm = {}
        fout = torch.zeros([1,10], device='cuda') + (output*0.7)
        for i, pred in enumerate(predictions):
            #sm[pred] = 0
            tot = 0.0
            expert = resnet20()
            for sub in SUBSET:
                if pred in sub and subset_flag[str(sub)] == True:

                    ###### Load the saved weights for the experts #####
                    wt = "./weights/rr/random_injection_erasing/res20_fmnist/rr_subset_" + str(sub) + ".pth.tar"

                    #wt = "./weights/latent_space_hardtraining/lp_subset_" + str(sub) + ".pth.tar"

                    wts = torch.load(wt)
                    expert.cuda()
                    expert.eval()
                    expert.load_state_dict(wts)
                    ############################

                    ### Inference part starts here ##########
                    output = F.softmax(expert(data))
                    #print (output)
                    #output = torch.sort(output, dim=1, descending=True)[0][0][0]
                    fout += output


                    #print (pred, target, output)
                    #sm[pred] += output.item()  #* trust_factor(len(sub), 2)
                    tot += 1
                    subset_flag[str(sub)] = False
        #fout = fout/tot
        #print ("Fout:",fout)
        prd = fout.data.max(1, keepdim=True)[1]
#

        if (prd == target.item()):
            correct  = correct + 1



       # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print ("\nThe routers performance: {:4f}".format(100.0*(tt.data.item()/len(test_loader.dataset))))
    print('EMNN (ours) accuracy: {:.4f}%)\n'.format(
    100. * correct / len(test_loader.dataset )))
    print ("Total time taken {:.2f}.".format(time.time()-start_time))
    delta = np.array(delta)
    fl = "./inference_result/fmnist_delta_resnet56_[4_3].txt"
    with open(fl, 'w') as f:
        for item in delta:
            f.write("%s\n" % item)

#if __name__ == "main":
test()

#model_size = sum( p.numel() for p in model.parameters() if p.requires_grad)

#print("The size of the expert model: {}M".format(model_size/1000000))
