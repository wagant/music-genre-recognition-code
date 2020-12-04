#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:05:56 2020

@author: piotrek
"""
import pandas as pd

class FMAMetadata():
    def __init__(self, path_to_tracks):
        if(path_to_tracks):
            self.tracks = pd.read_csv(path_to_tracks, 
                                      index_col=0, header=[0, 1])

    def get_top_genre(self, track_id):
        return self.tracks.loc[int(track_id), :]['track'].genre_top
    
    def get_set_split(self, track_id):
        return self.tracks.loc[int(track_id), :]['set'].split

    @staticmethod
    def save_to_file(handler, data, headers = None, separator = ','):
        if headers:
            handler.write(separator.join(headers) + '\n')
        for line in data:
            writeable = separator.join(line)
            handler.write(writeable + '\n')
    