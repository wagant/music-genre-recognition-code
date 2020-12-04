#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:18:17 2020

Preparing dataset -> from audio files to mel spectrograms using librosa package

@author: piotrek
"""

import librosa
import os
import numpy as np
import librosa.display
import matplotlib
import time

import warnings
warnings.filterwarnings('ignore')

matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATH_FILES = './fma_small/'

OUTPUT_PATH = 'spectrograms'

os.chdir('Documents/studia/mgr/master-thesis/')

def plot_spectrogram(waveform, sampling_rate):
    
    S = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
 
    
def save_img(dir, filename, failIfNotExists = False):
    
    if not os.path.exists(dir):
        if failIfNotExists:
            raise OSError('Provided directory does not exist ', dir)
        
        os.makedirs(dir)
    
    save_path = os.path.join(dir, filename.split('.')[0])
    plt.savefig(save_path + '.png')
    

def count_folders(dir):
    folders = [dirnames for _, dirnames, filenames in os.walk(dir)]
    
    # only top level folders
    return len(folders[0])
     
  
import shutil

def move_files(top_level_path):
    for folder in os.listdir(top_level_path):
        for file in os.listdir(os.path.join(top_level_path, folder)):
            print(file)  
            shutil.move(os.path.join(top_level_path, folder, file), os.path.join(top_level_path, file))


move_files('spectrograms/')

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
            y, sr  = librosa.load(os.path.join(PATH_FILES, folder, file))
        except Exception:
            print(f'problem with {file} occured')
            continue
        
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        ax.axis('off')
        
        plot_spectrogram(waveform=y, sampling_rate=sr)
        save_img(os.path.join(OUTPUT_PATH, folder), file)
        
        plt.close(fig)
        
    
    print('done printing ', folder)
    print(f'processed {processed}/{folders_count}')

print('done')

end = time.process_time()
print('it took ', end - start)
    
