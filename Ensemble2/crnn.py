#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 12:32:55 2020

@author: piotrek
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from print import Print

class CRNN(nn.Module):
    
    def __init__(self, hidden_size, num_layers, dropout, num_classes = 8, channels_in = 3):
        super(CRNN, self).__init__()
        self.num_classes = num_classes
        self.channels_in = channels_in
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layer1 = nn.Sequential(
             nn.Conv2d(channels_in, 64, kernel_size=3, stride=1, padding=2),
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
             nn.BatchNorm2d(128),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = nn.Sequential(
             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
             nn.BatchNorm2d(128),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2))
 
        self.lstm = nn.LSTM(9, hidden_size=self.hidden_size,
                      num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
 
        self.fc = nn.Linear(self.hidden_size, num_classes)

    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # output must be of size: batch_size, sequence length, n_features
        out = out.view(out.shape[0], -1, out.shape[-1])
        out, hidden = self.lstm(out)
        
        out = self.fc(out[:, -1, :])
        return out