#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 20:37:49 2020

@author: piotrek
"""


import seaborn as sns
import os
from utils import f1_score_mean, save_dict, precision_recall_mean
from linearnet import LinearNet
from models.crnn1d import CRNN1D
from crnn import CRNN
from torch import nn
import torch
from convnet import ConvNet
from fma_melspectrogram_loader import FMASpectrogramsDataset
from torchvision import transforms, utils
from transform import Rescale, RandomCrop, ToTensor, ToRGB, MeanSubstraction, MelspectrogramNormalize
from torch.utils.data import DataLoader, random_split
import fma_metadata
import matplotlib.pyplot as plt
import numpy as np

from level0.cnn1 import CNN1
from level0.cnn2 import CNN2
from level0.cnn3 import CNN3
from level0.cnn4 import CNN4

from utils import create_stacked_dataset


import torch.nn.functional as F


def print_models_layers(models):
    for model in models:
        for m in model.modules():
            print(m)


def save_models(models, path):
    for i, model in enumerate(models):
        target_dir = f'{path}/model_{i}.pt'
        torch.save(model.state_dict(), target_dir)


WORKING_DIRECTORY = 'Documents/studia/mgr/master-thesis'

if WORKING_DIRECTORY not in os.getcwd():
    os.chdir(WORKING_DIRECTORY)

dataset = FMASpectrogramsDataset(csv_file='track_mapping.txt', lookup_table='genres.txt', root_dir='spectrograms',
                                 transform=transforms.Compose([ToRGB(), Rescale((128, 128)),  MelspectrogramNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensor()]))

torch.manual_seed(100)
train_size = int(0.8 * len(dataset))
validation_size = int(0.1 * len(dataset))
test_size = len(dataset) - validation_size - train_size
train_dataset, test_dataset, validation_dataset = random_split(
    dataset, [train_size, validation_size, test_size])

torch.cuda.empty_cache()
# hyper
num_epochs = 30
num_classes = 8
batch_size = 16
learning_rate = 0.001


train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)


test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4)


validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=4)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


EXP_NAME = 'stacked_CNN_13'
models = []

# models.append(CNN1(num_classes).to(device))
# models.append(CNN2(num_classes).to(device))
# models.append(ConvNet(num_classes).to(device))
# models.append(CRNN(hidden_size=128, num_layers=3, dropout=0.2).to(device))
# models.append(CRNN1D(hidden_size=128, num_layers=2, dropout=0.2).to(device))
# models.append(CNN4(num_classes).to(device))


for m_i, model in enumerate(models):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_dataloader)
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            inputs, labels = data['image'].to(device), data['genre'].to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            inputs = inputs.float()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(
                    f'Model {m_i + 1} Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    torch.cuda.empty_cache()

for m_i, model in enumerate(models):
    model_cm = torch.zeros(num_classes, num_classes)
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):

            inputs, labels = data['image'].to(device), data['genre'].to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            inputs = inputs.float()

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            for t, p in zip(labels, predicted):
                model_cm[t, p] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        precision, recall = precision_recall_mean(model_cm)
        f1 = f1_score_mean(model_cm)
        print(f'Model {m_i + 1} accuracy {accuracy}')
        results = {
            'confusion_matrix': model_cm.tolist(),
            'accuracy': round(accuracy, 3),
            'precision': round(precision * 100, 3),
            'recall': round(recall * 100, 3),
            'f1 score': round(f1 * 100, 3)
        }
        save_dict(results, f'experiments/{EXP_NAME}/model_{m_i}_cm.txt')


save_models(models, f'experiments/{EXP_NAME}')

# experiments/stacked_CNN_1'
# models = []
# models.append(CNN2(num_classes).to(device))
# models.append(ConvNet(num_classes).to(device))
# # models.append(CRNN1D(hidden_size=128, num_layers=2, dropout=0.2).to(device))
# models.append(CRNN(hidden_size=128, num_layers=3, dropout=0.2).to(device))


# models[0].load_state_dict(torch.load(f'experiments/{EXP_NAME}/model_0.pt'))
# models[0].eval()
# models[1].load_state_dict(torch.load(f'experiments/{EXP_NAME}/model_1.pt'))
# models[1].eval()

# models[2].load_state_dict(torch.load(f'experiments/{EXP_NAME}/model_2.pt'))
# models[2].eval()

# # models[3].load_state_dict(torch.load(f'experiments/stacked_CNN_1/CRNN.pt'))


meta_lr = 0.0001

meta_learner = LinearNet(input_size=len(
    models) * num_classes, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(meta_learner.parameters(), lr=meta_lr)

# meta_learner.load_state_dict(torch.load(f'experiments/{EXP_NAME}/stacked_cnn_acc_51.25_n_models_3ep_5,bs_16'))

# # training meta learner

meta_epochs = 30
for epoch in range(meta_epochs):
    for i, data in enumerate(validation_dataloader):

        inputs, labels = data['image'].to(device), data['genre'].to(device)
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

torch.cuda.empty_cache()
# evaluate on test set


conf_matrix = torch.zeros(num_classes, num_classes)
# meta_learner.load_state_dict(torch.load(
#     'experiments/stacked_CNN_1/stacked_cnn_acc_43.38_n_models_3ep_5,bs_16'))

meta_learner.eval()

VOTING = False

total = 0
correct = 0
with torch.no_grad():
    for i, data in enumerate(test_dataloader):

        inputs, labels = data['image'].to(device), data['genre'].to(device)
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

    accuracy = 100 * correct / total
    f1_score = f1_score_mean(conf_matrix)
    precision, recall = precision_recall_mean(conf_matrix)

    print("accuracy", accuracy)
    print("f1 score", f1_score)

    model_path = f'stacked_cnn_acc_{round(accuracy,2)}_n_models_{len(models)}ep_{num_epochs},bs_{batch_size}'
    results_meta = {
        "confusion_matrix": conf_matrix.tolist(),
        "f1_score": round(f1_score * 100, 3),
        "accuracy": round(accuracy, 3),
        "precision": round(precision * 100, 3),
        "recall": round(recall * 100, 3)
    }

    save_dict(results_meta,
              f'experiments/{EXP_NAME}/meta_learner_results_voting_{VOTING}.txt')
