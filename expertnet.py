
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:07:51 2019

@author: Intisar

This is a program to segregate the dataset into confusable class.


"""

from __future__ import print_function
import torchvision.models as models
import torch.nn as nn
from resnet import resnet56
############## THE EXPERT MODULES #############################################


class ExpertNet(nn.Module):

    def __init__(self):
        super(ExpertNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block1 = resnet56().layer1
        self.block2 = resnet56().layer2
        
        #self.block1 = models.resnet50().layer1
        #self.block2 = models.resnet50().layer2
        # avg pooling to global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
                nn.Linear(in_features=32, out_features=512, bias=True),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p = 0.5),
                nn.Linear(512, 10),
                )
                

    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block2(self.block1(x))
        x = self.avgpool(x)
        print (x.shape)
        x = x.view(x.size(0), x.size(1))
        out = self.fc(x)
        return out
        
##########################################################################################
        
### Utility to test the expert module    
        
if __name__ == "__main__":
    r = models.resnet18()
    print (r)
    input = Variable(torch.FloatTensor(8, 3, 64, 64))
    model = ExpertNet()
    out = model(input)
    print (out.shape)   
    print (model)
    model_size = sum( p.numel() for p in model.parameters() if p.requires_grad)

    print("The size of the Teacher Model: {}M".format(model_size))

