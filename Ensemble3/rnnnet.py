#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:51:21 2020

@author: piotrek
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, batch_size, dropout=0, num_classes=8):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_classes = num_classes

        super(RNNNet, self).__init__()
        # input of size batch_size seq_len, input_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, dropout=self.dropout, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, num_classes)

    def init_hidden(self):
        # returns (hidden state, cell state)
        return (torch.randn(self.num_layers, self.batch_size, self.hidden_size),
                torch.randn(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, x):
        out, hidden = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
