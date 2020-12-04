#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:57:59 2020

Utils tools for files manipulation and preprocessed data validation

@author: piotrek
"""


import os
import shutil
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def move_files(top_level_path):
    for folder in os.listdir(top_level_path):
        for file in os.listdir(os.path.join(top_level_path, folder)):
            print(file)
            shutil.move(os.path.join(top_level_path, folder, file),
                        os.path.join(top_level_path, file))


def compare_files_count(path1, path2):

    path1_files = []
    for subdir, dirs, files in os.walk(path1):
        path1_files = list(map(lambda file: file.split('.')[0], files))

    path2_files = []

    for subdir, dirs, files in os.walk(path2):
        path2_files = list(map(lambda file: file.split('.')[0], files))

    return path1_files, path2_files


def substract_lists(bigger, smaller):
    return [item for item in bigger if item not in smaller]


def folders_to_recomp(missing_files, top_level):
    corrupted_folders = {}
    for folder in os.listdir(top_level):
        missing = []
        for file in os.listdir(os.path.join(top_level, folder)):
            name = file.split('.')[0]
            if name in missing_files:
                missing.append(name)
        if len(missing) > 0:
            corrupted_folders[folder] = missing

    return corrupted_folders


def validate_files(path):
    corrupted_files = []
    for file in os.listdir(path):
        name = file.split('.')[0]

        filename = os.path.join(path, file)
        try:
            with open(filename) as f:
                mfcc_data = json.load(f)
        except:
            corrupted_files.append(name)

    return corrupted_files


import pandas as pd

# asd = pd.read_csv('track_mapping.txt', dtype='str')

# file = asd[asd['track_id'] == '115176']
# genre = file.genre.values[0]

def exclude_genres(files_path, genres, target_path, mapping_path): 
    genres_to_add = []
    tracks_to_add = []
    lookup_table = pd.read_csv(mapping_path, dtype='str')
    for file in os.listdir(os.path.join(files_path)):
        genre = lookup_table[lookup_table['track_id'] == file.split('.')[0]].genre.values[0]
        if genre in genres:
            continue
        shutil.copy(os.path.join(files_path, file), os.path.join(target_path, file))
        tracks_to_add.append(file.split('.')[0])
        genres_to_add.append(genre)
    new_mapping = pd.DataFrame(list(zip(tracks_to_add, genres_to_add)), columns=['track_id','genre'])
    new_mapping.to_csv('track_mapping_excluded.txt', index=False)
    print('saved')


import torch
import torch.nn.functional as F
# stacking
# batch size 
def create_stacked_dataset(models, input_batch, num_classes, use_probs = False):
    # input batch: batch_size x channels x height x width
    dataset = torch.empty(input_batch.shape[0], num_classes, len(models))
   
    for i, model in enumerate(models):
        model.train()
        # make prediction 
        # return prob value for each class
        # add record to the datastet
        # so in this case the dimensions would be (batch_size,  n_classes (8), len(models))
        
        outputs = model(input_batch)
        
        if use_probs:
            outputs = F.softmax(outputs)
        
        dataset[:,:, i] = outputs
        
    
    return dataset
    # flatten the last 2 dimensions
    # dataset.view(dataset.shape[0], -1)


def move_files_split(top_level_path, mapping_path):
    mapping = pd.read_csv(mapping_path, dtype='str')
    
    split_col = mapping.columns.values[-1]
    
    splits = list(mapping[split_col].unique())
    
    # create if do not exist
    for split in splits:
        path = os.path.join(top_level_path, split)
        if not os.path.exists(path):
            os.mkdir(path)
            
    for file in os.listdir(top_level_path):
        # do not process folders
        if file in splits:
            continue
        found = mapping[mapping['track_id'] == file.split('.')[0]]
        if found is None:
            continue
        song_split = found[split_col].values[0]
        
        shutil.move(os.path.join(top_level_path, file), os.path.join(top_level_path, song_split, file))


def f1_score_mean(confusion_matrix):
    # predicted x true
    n_classes = confusion_matrix.shape[0]
    f1 = []
    for c in range(n_classes):
        recall = confusion_matrix[c,c] / confusion_matrix[:,c].sum()
        precision = confusion_matrix[c,c] / confusion_matrix[c, :].sum()
        f1_score = 2 * (precision * recall)/(precision + recall)
        f1.append(f1_score)
    
    return (sum(f1) / len(f1)).item()


def precision_recall_mean(confusion_matrix):
    # predicted x true
    n_classes = confusion_matrix.shape[0]
    f1 = []
    precisions = []
    recalls = []
    for c in range(n_classes):
        recall = confusion_matrix[c,c] / confusion_matrix[:,c].sum()
        recalls.append(recall)
        
        precision = confusion_matrix[c,c] / confusion_matrix[c, :].sum()
        precisions.append(precision)
    
    return (sum(precisions) / len(precisions)).item(), (sum(recalls) / len(recalls)).item()


def save_dict(obj, path):
    with open(path, 'w+') as f:
        f.write(json.dumps(obj))


def accuracy(cm):
    return (cm.diag().sum()/cm.sum()).item()


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Reds')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.title('Confusion matrix', fontsize=22)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

        

###             

WORKING_DIRECTORY = 'Documents/studia/mgr/master-thesis'

if WORKING_DIRECTORY not in os.getcwd():
    os.chdir(WORKING_DIRECTORY)


# move_files('mfccs')


# path1 = 'mfccs'
# path2 = 'spectrograms'

# list1, list2 = compare_files_count(path1, path2)

# diff = substract_lists(list2, list1)

# corrupted_folders = folders_to_recomp(diff, 'fma_small')

# corrupted_files = validate_files(path1)

# corrupted_files = list(map(lambda f: f.split('.')[0], corrupted))

# corrupted_folders = folders_to_recomp(corrupted_files, 'fma_small')


# path3 = 'excluded_spectrograms'

# exclude_genres(path2, ['Hip-Hop', 'Experimental'], path3, 'track_mapping.txt')

# move_files_split(path2, 'track_mapping.txt')

# move_files(path2)



# cm = torch.Tensor([[77.0, 2.0, 1.0, 2.0, 2.0, 9.0, 3.0, 1.0], 
#                    [2.0, 65.0, 5.0, 5.0, 13.0, 2.0, 6.0, 2.0], 
#                    [6.0, 7.0, 42.0, 5.0, 18.0, 11.0, 6.0, 12.0], 
#                    [13.0, 4.0, 2.0, 52.0, 1.0, 4.0, 6.0, 3.0], 
#                    [3.0, 11.0, 15.0, 1.0, 57.0, 2.0, 2.0, 10.0], 
#                    [17.0, 0.0, 4.0, 1.0, 12.0, 68.0, 5.0, 2.0], 
#                    [11.0, 14.0, 3.0, 17.0, 7.0, 11.0, 18.0, 12.0], 
#                    [1.0, 11.0, 6.0, 5.0, 0.0, 4.0, 8.0, 71.0]])

# cm = cm.int().numpy()

# labels = ['Hip-Hop', 'Folk', 'Experimental', 'International',
# 'Instrumental','Electronic','Pop','Rock']


# print_confusion_matrix(cm, labels)
