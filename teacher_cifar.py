from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from resnet import resnet20
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import torchvision
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import  SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
import transforms
# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=bool, default=True, metavar='N',
                    help='resume from the last weights')

# Random Erasing
parser.add_argument('--p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.3, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.2, type=float, help='aspect of erasing area')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def accuracy_topk(output, target, topk=(3,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, correct_k


def doPlot(val_scores):

    scores = []
    scores = [ h.cpu().numpy() for h in val_scores]
    print (scores)
    plt.title("Teacher CNN")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, args.epochs+1), scores)
    plt.ylim( (0,1.) )
    plt.xticks(np.arange(1, args.epochs+1, 1.0))
    plt.show()


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




train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=True, download=True,
                   transform=transform_train),
    batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=False, transform=transform_test),
    batch_size=args.test_batch_size, shuffle=True)

###############################################################################




# LOAD THE RESENT-50 with trained on CIFAR-10 (PRETRAINED ON IMAGENET)
# NO data increment.

###############################################################################
from resnet import  resnet20
model = resnet20()
# model = resnet20()
#model = torchvision.models.resnet50(pretrained=True)
#model.linear = nn.Linear(in_features=64, out_features=100, bias=True)
#checkpoint = torch.load("teacher_MLP_test_eresnet56_best_archi.pth.tar")
# checkpoint = torch.load('./weights/best_so_far_res56.pth.tar')
#checkpoint = torch.load("./weights/suSan.pth.tar")
checkpoint = torch.load("./resnet20_fmnist.pth.tar")

#router_resnet20_all_class.pth.tar
model.load_state_dict(checkpoint)


if args.cuda:
    model.cuda()


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
def train(epoch, model):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.softmax(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        #pred = output.data.max(1, keepdim=True)[1]

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.item()))


def train_evaluate(model):
    model.eval()
    train_loss = 0
    correct = 0
    now_correct = 0
    plt_score_val = []
    global best_correct
    for data, target in train_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        train_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct.double() / len(train_loader.dataset)))
    now_correct = correct.double()/len(train_loader.dataset)
    if best_correct < now_correct:
        best_correct = now_correct
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, 'resnet20_fmnist.pth.tar')
    return now_correct

best_correct = -999
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    now_correct = 0
    c = 0
    sml = []
    global best_correct
    for data, target in test_loader:
#        if c == 200:
#            break
#        c = c+ 1
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        output = F.softmax(output)
        test_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        p = output.data.max(1, keepdim=True)[0]
        sml.append(p.item())
        pred = torch.argsort(output, dim=1, descending=True)[0:, 0]
        pred2 = torch.argsort(output, dim=1, descending=True)[0:, 1] # top 2
        pred3 = torch.argsort(output, dim=1, descending=True)[0:, 2] # top 3

        #soft_max_sorted = torch.sort(output, dim=1, descending=True)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        #correct += pred2.eq(target.data.view_as(pred2)).cpu().sum()
        #correct += pred3.eq(target.data.view_as(pred3)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, TOP 1 Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
         100. * correct.double() / len(test_loader.dataset) ))
    
    fl = "softmaxFmnist.txt"
    with open(fl, 'w') as f:
        for item in sml:
            f.write("%s\n" % item)
    return (100. * correct.double() / len(test_loader.dataset))

#    if best_correct < now_correct:
#        best_correct = now_correct
#        best_model_wts = copy.deepcopy(model.state_dict())
#        torch.save(best_model_wts, 'resnet56_cifar_10_random_erasing.pth.tar')
#    return now_correct
#
#    report = classification_report(y_pred, y_truth)
#    print (confusion_matrix(y_pred, y_truth))
#    print (report)

scores = np.array([])
args.epochs =  1
val_scores = []
for epoch in range(1, args.epochs + 1):
     # train(epoch, model)
     # score = train_evaluate(model)
     # val_scores.append(score)
     # start_time = time.time()
     # scores = np.append(scores, test(model))
     # scheduler.step()
     test(model)
print ("The mean accuracy: {}".format(np.mean(scores)))
print ("The STD : {}: ".format(np.std(scores)))


#doPlot(val_scores)

with open('fmnist_resnet20_result.txt', 'w') as f:
   for item in val_scores:
       f.write("%s\n" % item)

model_size = sum( p.numel() for p in model.parameters() if p.requires_grad)

print("The size of the Teacher Model: {}".format(model_size))


#torch.save(best_model_wts, 'teacher_MLP.pth.tar')
# the_model = Net()
# the_model.load_state_dict(torch.load('teacher_MLP.pth.tar'))

# test(the_model)
# for data, target in test_loader:
#     data, target = Variable(data, volatile=True), Variable(target)
#     teacher_out = the_model(data)
# print(teacher_out)
#print("--- %s seconds ---" % (time.time() - start_time))

#
#def show_batch(batch):
#    im = torchvision.utils.make_grid(batch)
#    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
#    plt.show()
