#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 12:32:55 2020

@author: piotrek
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN1D(nn.Module):
    
    def __init__(self, hidden_size, num_layers, dropout, num_classes = 8, channels_in = 3):
        super(CRNN1D, self).__init__()
        self.num_classes = num_classes
        self.channels_in = channels_in
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer1 = nn.Sequential(
             nn.Conv1d(channels_in, 128, kernel_size=5, stride=1, padding=2),
             nn.BatchNorm1d(128),
             nn.ReLU(),
             nn.MaxPool1d(kernel_size=2, stride=2)
             )

        self.layer2 = nn.Sequential(
             nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
             nn.BatchNorm1d(128),
             nn.ReLU(),
             nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
              nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm1d(64),
              nn.ReLU(),
              nn.MaxPool1d(kernel_size=2, stride=2))
        
        # self.layer4 = nn.Sequential(
        #       nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
        #       nn.BatchNorm2d(64),
        #       nn.ReLU(),
        #       nn.MaxPool2d(kernel_size=2, stride=2))
 
        self.lstm = nn.LSTM(3072, hidden_size=self.hidden_size,
                      num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
 
        self.fc1 = nn.Linear(self.hidden_size, num_classes)

        # self.fc2 = nn.Linear(32, num_classes)

    
    def forward(self, x):
        # out = self.layer1(x)
        out = self.layer1(x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        
        # output must be of size: batch_size, sequence length, n_features
        out = out.view(out.shape[0], -1, out.shape[-1])
        out, hidden = self.lstm(out)
        out = self.fc1(out[:, -1, :])
        # out = self.fc2(out)
        return out