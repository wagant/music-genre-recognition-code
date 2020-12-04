#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:27:44 2020

@author: piotrek
"""


import os
from fma_metadata import FMAMetadata

'''
flow:
    1. load file into memory
    2. get track id from the name
    3. based on track id get the top level genre and set split
    4. save the track, genre and set to a file
'''

FILES_PATH = 'mfccs'
CSV_PATH = 'fma_metadata/tracks.csv'

metadata = FMAMetadata(CSV_PATH)

classes = {}


WORKING_DIRECTORY = 'Documents/studia/mgr/master-thesis'

if WORKING_DIRECTORY not in os.getcwd():
    os.chdir(WORKING_DIRECTORY)


sets = ['training', 'validation', 'test']

for set_name in sets:
    classes[set_name] = []
    for img in os.listdir(FILES_PATH):
        track_id = img.split('.')[0]
        # to trim 0s
        
        genre = metadata.get_top_genre(int(track_id))
        set_split = metadata.get_set_split(int(track_id))
        if set_split == set_name:
            classes[set_name].append((track_id, genre, set_split))
    
    # every set to another file
    with open(f'track_mapping_{set_name}.txt', 'w') as file:
        FMAMetadata.save_to_file(file, classes[set_name], ['track_id', 'genre', 'split_set'])
