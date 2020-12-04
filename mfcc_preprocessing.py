#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:46:33 2020

@author: piotrek
"""


import os
import numpy as np


import time

import pandas as pd
import json
import librosa

import warnings
warnings.filterwarnings('ignore')

PATH_FILES = './fma_small/'
OUTPUT_PATH = 'mfccs'
SAMPLE_RATE = 44100 // 4
N_MFCC = 13
GENRES_MAPPING = 'genres.txt'
TRACK_MAPPING = 'meta_genres.txt'


WORKING_DIRECTORY = 'Documents/studia/mgr/master-thesis'

if WORKING_DIRECTORY not in os.getcwd():
    os.chdir(WORKING_DIRECTORY)

genres_mapping = pd.read_csv(GENRES_MAPPING)
genres_mapping = dict(zip(genres_mapping.genre, genres_mapping.id))
track_genre_mapping = pd.read_csv(TRACK_MAPPING, dtype={'track_id': str})
track_genre_mapping = dict(zip(track_genre_mapping.track_id, track_genre_mapping.genre)) 

def count_folders(dir):
    folders = [dirnames for _, dirnames, filenames in os.walk(dir)]
    
    # only top level folders
    return len(folders[0])
     

data = { "mfccs": [] }

start = time.process_time()


processed = 0
folders_count = count_folders(PATH_FILES)
for folder in os.listdir(PATH_FILES):
    processed += 1
    if os.path.exists(os.path.join(OUTPUT_PATH, folder)): # in ['018', '044', '066', '072', '076', '090']:
        print('skipped folder ', folder)
        continue
    print('processing ', folder)
    for file in os.listdir(os.path.join(PATH_FILES, folder)):
            
        try:
            # waveform, sr  = torchaudio.load(os.path.join(PATH_FILES, folder, file))
            waveform, _ = librosa.load(os.path.join(PATH_FILES, folder, file), sr=SAMPLE_RATE)
        except Exception:
            print(f'problem with {file} occured')
            continue
        
        # get a tensor of mfcc
        mfcc = librosa.feature.mfcc(waveform, sr = SAMPLE_RATE, n_mfcc = N_MFCC)
        # mfcc = torchaudio.transforms.MFCC(sample_rate = SAMPLE_RATE, n_mfcc = N_MFCC)(waveform)
        mfcc = mfcc.T.tolist()
        
        track_id = file.split('.')[0]
        genre =  track_genre_mapping.get(track_id)
        if genre:    
            genre_mapped = genres_mapping.get(genre)
        else:
            continue
        
        # data["track_ids"].append(track_id)
        data["mfccs"] = mfcc
        # data["labels"].append(genre_mapped)
 

        outdir = os.path.join(OUTPUT_PATH, folder)
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        with open(os.path.join(outdir, track_id + '.json' ), 'w') as f:
            json.dump(data, f, indent=4)
    
    print('done printing ', folder)
    print(f'processed {processed}/{folders_count}')

    

print('done')

end = time.process_time()
print('it took ', end - start)
