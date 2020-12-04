#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 21:00:25 2020

@author: piotrek
"""

from utils import f1_score_mean, accuracy, save_dict, precision_recall_mean
from torch import nn
from rnnnet import RNNNet
import torch

from fma_mfcc_loader import FMAMfccDataset

from torch.utils.data import DataLoader, random_split

from transform import ListToTensor

import os
from torchvision import transforms

def train_model(model, dataloader, num_epochs, criterion, optimizer, device):
    
    # Train the model
    total_step = len(dataloader)
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            inputs, labels = data['mfcc'].to(device), data['genre'].to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            model.zero_grad()

            # Forward pass
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
                
    return model

def test_model(model, dataloader, device):
    cm = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        h = model.init_hidden()
        for i, data in enumerate(dataloader):
    
            inputs, labels = data['mfcc'].to(device), data['genre'].to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
        
            outputs = model(inputs)
        
            _, predicted = torch.max(outputs.data, 1)
            
            for t, p in zip(labels, predicted):
                cm[t, p] += 1
    
    return cm


WORKING_DIRECTORY = 'Documents/studia/mgr/master-thesis'

if WORKING_DIRECTORY not in os.getcwd():
    os.chdir(WORKING_DIRECTORY)

torch.manual_seed(100)

transform = transforms.Compose([ListToTensor()])
dataset = FMAMfccDataset(csv_file='track_mapping.txt',
                         lookup_table='genres.txt', root_dir='mfccs', transform=transform)



train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# train_dataset = FMAMfccDataset(csv_file='track_mapping_training.txt', lookup_table='genres.txt', root_dir='mfccs/training',
#                                  transform=transform)


# test_dataset = FMAMfccDataset(csv_file='track_mapping_test.txt', lookup_table='genres.txt', root_dir='mfccs/test',
#                                  transform=transform)



# hyper
num_epochs = 5
num_classes = 8
batch_size = 16
learning_rate = 0.001
hidden_size = 256
num_layers = 3
dropout = 0.2


train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, drop_last=True)


test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4, drop_last=True)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

n_coeffs = 13

rnn_experiments = {
   'RNNx2x256v2': RNNNet(input_size=n_coeffs, hidden_size=256, num_layers=2,
                       batch_size=batch_size, dropout=dropout, num_classes=num_classes).to(device),
    'RNNx4x128v2': RNNNet(input_size=n_coeffs, hidden_size=128, num_layers=4,
                        batch_size=batch_size, dropout=dropout, num_classes=num_classes).to(device),
    }



for model_name, model in rnn_experiments.items():
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model = train_model(model, train_dataloader, num_epochs, criterion, optimizer, device)
    torch.cuda.empty_cache()
    model.eval()
    
    cm_rnn = test_model(model, test_dataloader, device)

    acc = accuracy(cm_rnn)
    f1_score = f1_score_mean(cm_rnn)
    precision, recall = precision_recall_mean(cm_rnn)
    
    hyper_parameters = {
        'epochs': num_epochs,
        'batch_size': batch_size,
        'lr': learning_rate,
        'dropout': dropout
        }
    
    results = {
        'confusion_matrix': cm_rnn.tolist(),
        'accuracy': acc,
        'f1 score': f1_score,
        'precision': precision,
        'recall': recall
        }
    
    save_dict(results, f'experiments/{model_name}/results.txt')
    save_dict(hyper_parameters, f'experiments/{model_name}/hyper.txt')
    
    
    model_path = f'experiments/{model_name}/{model_name}.pt'
    torch.save(model.state_dict(), model_path)
    
    print('f1 score', f1_score)
    print(model_name, 'done')


