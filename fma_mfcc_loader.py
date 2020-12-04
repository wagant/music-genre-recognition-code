#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:24:25 2020

@author: piotrek
"""

import os
import torch
import pandas as pd

from skimage import io
from torch.utils.data import Dataset
import numpy as np
import json


import warnings
warnings.filterwarnings("ignore")

class FMAMfccDataset(Dataset):
    """ Free Music Spectrogram Audio dataset loader """
    
    def __init__(self, csv_file, root_dir, lookup_table, transform=None):
        """     
        Parameters
        ----------
        csv_file : string
            path to metadata file. (cols: filename, genre)
        root_dir : string
            top level dir with mel spectrograms.
        transform : callabe, optional
            transform callable class. The default is None.

        Returns
        -------
        Dataloader instance

        """
        self.transform = transform
        self.genres = pd.read_csv(csv_file, dtype='str')
        self.root_dir = root_dir
        self.lookup = pd.read_csv(lookup_table, dtype='str')
        
 
    def __len__(self):
        return len(self.genres)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = os.path.join(self.root_dir, str(self.genres.loc[idx].track_id) +'.json')
        
        with open(filename, 'r') as file:
            mfcc_data = json.load(file)
        
        genre = self.genres.loc[idx].genre
        
        # map  to label id from lookup  table
        if not self.lookup.empty:   
            idx = self.lookup.loc[self.lookup['genre'] == genre].index[0]
            genre =  np.asarray(float(self.lookup.loc[idx].id))
        
        sample = {
            'mfcc': mfcc_data['mfccs'],
            'genre': genre
            }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        