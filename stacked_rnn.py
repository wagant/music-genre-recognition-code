#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 12:36:52 2020

@author: piotrek
"""


from linearnet import LinearNet
from torch import nn
import torch
from convnet import ConvNet
from torchvision import transforms, utils
from torch.utils.data import DataLoader, random_split
import fma_metadata
import matplotlib.pyplot as plt
import os
from transform import ListToTensor

from rnnnet import RNNNet
from fma_mfcc_loader import FMAMfccDataset

import torch.nn.functional as F
from torchvision import transforms

from utils import create_stacked_dataset, f1_score_mean, precision_recall_mean, save_dict

import numpy as np

WORKING_DIRECTORY = 'Documents/studia/mgr/master-thesis'

if WORKING_DIRECTORY not in os.getcwd():
    os.chdir(WORKING_DIRECTORY)


EXP_NAME = 'stacked_RNN_2'

transform = transforms.Compose([ListToTensor()])
dataset = FMAMfccDataset(csv_file='track_mapping.txt',
                         lookup_table='genres.txt', root_dir='mfccs', transform=transform)

torch.manual_seed(100)

# hyper
num_epochs = 5
num_classes = 8
batch_size = 16
learning_rate = 0.001
hidden_size = 256
num_layers = 3
dropout = 0.2


train_size = int(0.8 * len(dataset))
test_size = int(0.1 * len(dataset))
validation_size = len(dataset) - train_size - test_size

train_dataset, test_dataset, validation_dataset = random_split(
    dataset, [train_size, test_size, validation_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)


test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4)


validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=4)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


n_coeffs = 13


models = []
models.append(RNNNet(input_size=n_coeffs, hidden_size=256, num_layers=2,
                     batch_size=batch_size, dropout=0.2, num_classes=num_classes).to(device))
models.append(RNNNet(input_size=n_coeffs, hidden_size=128, num_layers=3,
                     batch_size=batch_size, dropout=0.1, num_classes=num_classes).to(device))
models.append(RNNNet(input_size=n_coeffs, hidden_size=256, num_layers=3,
                     batch_size=batch_size, dropout=0.3, num_classes=num_classes).to(device))


for m_i, model in enumerate(models):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_dataloader)
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
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
                    f'Model {m_i} Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')


with torch.no_grad():
    for m_i, model in enumerate(models):
        total = 0
        correct = 0
        h = model.init_hidden()

        model_cm = torch.zeros(num_classes, num_classes)

        for i, data in enumerate(test_dataloader):

            inputs, labels = data['mfcc'].to(device), data['genre'].to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            for t, p in zip(labels, predicted):
                model_cm[t, p] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        torch.save(model.state_dict(),
                   f'experiments/{EXP_NAME}/model_{m_i}.pt')

        accuracy = correct / total
        print('hidden size', hidden_size)
        print(f'model {m_i} accuracy {accuracy}')

        f1_score = f1_score_mean(model_cm)
        precision, recall = precision_recall_mean(model_cm)
        results = {
            'confusion_matrix': model_cm.tolist(),
            'accuracy': round(accuracy * 100, 3),
            'f1_score': round(f1_score * 100, 3),
            "precision": round(precision * 100, 3),
            "recall": round(recall * 100, 3)
        }
        save_dict(results, f'experiments/{EXP_NAME}/model_{m_i}_cm.txt')


meta_lr = 0.0001

meta_learner = LinearNet(input_size=len(
    models) * num_classes, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(meta_learner.parameters(), lr=meta_lr)


meta_epochs = 30
for epoch in range(meta_epochs):

    for i, data in enumerate(validation_dataloader):

        inputs, labels = data['mfcc'].to(device), data['genre'].to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        inputs = inputs.float()
        stack_train_X = create_stacked_dataset(
            models, inputs, num_classes, use_probs=True).to(device)

        outputs = meta_learner(stack_train_X.view(stack_train_X.shape[0], -1))

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (i+1) % 10 == 0:
        print(
            f' Epoch [{epoch+1}/{meta_epochs}], Loss: {loss.item():.4f}')


VOTING = True
torch.cuda.empty_cache()
# evaluate on test set

total = 0
correct = 0
with torch.no_grad():
    conf_matrix = torch.zeros(num_classes, num_classes)
    for i, data in enumerate(test_dataloader):

        inputs, labels = data['mfcc'].to(device), data['genre'].to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        inputs = inputs.float()
        stack_test_X = create_stacked_dataset(
            models, inputs, num_classes, use_probs=True).to(device)

        outputs_final = meta_learner(
            stack_test_X.view(stack_test_X.shape[0], -1))

        _, predicted = torch.max(outputs_final.data, 1)

        final_predicted = torch.zeros(
            len(outputs_final), dtype=torch.long).to(device)

        if VOTING:
            # majority voting or max from meta-learner

            # stack_test_X shape: torch.Size([14, 8, 3])
            models_remarks = torch.zeros(
                stack_test_X.shape[0], stack_test_X.shape[2])
            # shape 14, 3
            for z in range(stack_test_X.shape[2]):

                # find the max value for each class for each model
                max_tensor = torch.max(stack_test_X[:, :, z], dim=1)
                predicted_classes = max_tensor.indices

                # save best values to models_remarks
                models_remarks[:, z] = predicted_classes

            # iterate through number of samples in minibatch
            # decide what should be the final prediction
            for n in range(models_remarks.shape[0]):
                temp = models_remarks[n]

                unique, counts = np.unique(temp, return_counts=True)

                # if there is more than 1 occurence, find for what value
                occurences = np.where(counts > 1, unique, 0)
                if np.count_nonzero(occurences) != 0:
                    # index is also the class
                    idx_max = np.unique(occurences.max())
                    idx_max = torch.tensor(
                        idx_max, dtype=torch.long, device=device)
                    final_predicted[n] = idx_max
                else:
                    final_predicted[n] = predicted[n]
        else:
            final_predicted = predicted

        total += labels.size(0)
        correct += (final_predicted == labels).sum().item()

        for t, p in zip(labels, final_predicted):
            conf_matrix[t, p] += 1

    f1_score = f1_score_mean(conf_matrix)
    precision, recall = precision_recall_mean(conf_matrix)

    accuracy = 100 * correct / total
    print(accuracy)

    results_meta = {
        "confusion_matrix": conf_matrix.tolist(),
        'accuracy': round(accuracy, 3),
        'precision': round(precision * 100, 3),
        'recall': round(recall * 100, 3),
        'f1 score': round(f1_score * 100, 3)
    }

    save_dict(results_meta, f'experiments/{EXP_NAME}/meta_learner_results.txt')

    model_path = f'experiments/{EXP_NAME}/stacked_rnn_acc_{round(accuracy,2)}_n_models_{len(models)}ep_{num_epochs},bs_{batch_size}_voting={VOTING}'
    torch.save(meta_learner.state_dict(), model_path)
