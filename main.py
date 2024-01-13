#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:15:03 2024

@author: akapoor
"""

import numpy as np
import torch
import sys
import os
os.chdir('/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_old_version/USA_5207_Analysis/')
# sys.path.append('/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR/USA_5207_Analysis/')
from util import WavtoSpec, Tweetyclr
import pickle
import umap
import matplotlib.pyplot as plt

# TODO: Include capacity for colorizing by multiple days and loading data from multiple days. 

# =============================================================================
#     # Set data parameters
# =============================================================================

source_dir = '/home/akapoor/Dropbox (University of Oregon)/USA5323_songs/'
dest_dir = '/home/akapoor/Dropbox (University of Oregon)/AK_RHV_Analysis/USA5323/Python_Files'

# spectrogram_creator = WavtoSpec(source_dir, dest_dir)
# spectrogram_creator.process_directory()

# This is the upstream location where analysis results are stored.
analysis_path = '/home/akapoor/Dropbox (University of Oregon)/AK_RHV_Analysis/USA5323/UMAP_Analysis/'

# Parameters we set
num_spec = 10 # The number of spectrograms for analysis 
window_size = 100 # The window size for UMAP 
stride = 10 # Stride size for analysis 

# Define the folder name. THe results will be directly stored here. 
folder_name = f'{analysis_path}Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}' # 

lowThresh = 500
highThresh = 7000
# masking_freq_tuple = (lowThresh, highThresh) # Apply low and high pass filtering 
spec_dim_tuple = (window_size, 151) # dimensions of spec slices that are passed into UMAP 

# Set parameters
bird_dir = dest_dir # LOcation of the NPZ files 

files = os.listdir(bird_dir)
all_songs_data = [f'{bird_dir}/{element}' for element in files if '.npz' in element] # Get the file paths of each numpy file from Yarden's data
all_songs_data.sort() # Sort spectrograms chronologically 


# Create the folder if it doesn't already exist
if not os.path.exists(folder_name+"/Plots/Window_Plots"):
    os.makedirs(folder_name+"/Plots/Window_Plots")
    print(f'Folder "{folder_name}" created successfully.')
else:
    print(f'Folder "{folder_name}" already exists.')
    
    
dat = np.load(all_songs_data[0])
spec = dat['s']
times = dat['t']
frequencies = dat['f']
labels = dat['labels'] # All values are 0. We can change the label values to indicate the day or pre vs post lesion 
labels = labels.T

plt.figure()
plt.pcolormesh(times, frequencies, spec, cmap='jet')
plt.show()

# For each spectrogram we will extract
# 1. Each timepoint's syllable label
# 2. The spectrogram itself
stacked_labels = [] 
stacked_specs = []
for i in np.arange(num_spec):
    # Extract the data within the numpy file. We will use this to create the spectrogram
    dat = np.load(all_songs_data[i])
    spec = dat['s']
    times = dat['t']
    frequencies = dat['f']
    labels = dat['labels'] # Labels that will be used for color coding UMAP points
    labels = labels.T


    # Apply high and low frequency thresholds to get a subsetted spectrogram
    mask = (frequencies<highThresh)&(frequencies>lowThresh)
    masked_frequencies = frequencies[mask]

    subsetted_spec = spec[mask.reshape(mask.shape[0],),:]
    
    stacked_labels.append(labels)
    stacked_specs.append(subsetted_spec)

    
stacked_specs = np.concatenate((stacked_specs), axis = 1)
stacked_labels = np.concatenate((stacked_labels), axis = 0)

# Get a list of unique categories (syllable labels)
unique_categories = np.unique(stacked_labels) # Efficient colorizing 

# Create a dictionary that maps categories to random colors
category_colors = {category: np.random.rand(3,) for category in unique_categories}


# =============================================================================
#     # Code for the windowing procedure 
# =============================================================================
spec_for_analysis = stacked_specs.T
window_labels_arr = []
embedding_arr = []
# Find the exact sampling frequency (the time in miliseconds between one pixel [timepoint] and another pixel)
dx = np.diff(times)[0,0]

# We will now extract each mini-spectrogram from the full spectrogram
stacked_windows = []
# Find the syllable labels for each mini-spectrogram
stacked_labels_for_window = []
# Find the mini-spectrograms onset and ending times 
stacked_window_times = []

# The below for-loop will find each mini-spectrogram (window) and populate the empty lists we defined above.
for i in range(0, spec_for_analysis.shape[0] - window_size + 1, stride):
    # Find the window
    window = spec_for_analysis[i:i + window_size, :]
    # Get the window onset and ending times
    window_times = dx*np.arange(i, i + window_size)
    # We will flatten the window to be a 1D vector
    window = window.reshape(1, window.shape[0]*window.shape[1])
    # Extract the syllable labels for the window
    labels_for_window = stacked_labels[i:i+window_size, :]
    # Reshape the syllable labels for the window into a 1D array
    labels_for_window = labels_for_window.reshape(1, labels_for_window.shape[0]*labels_for_window.shape[1])
    # Populate the empty lists defined above
    stacked_windows.append(window)
    stacked_labels_for_window.append(labels_for_window)
    stacked_window_times.append(window_times)

# Convert the populated lists into a stacked numpy array
stacked_windows = np.stack(stacked_windows, axis = 0)
stacked_windows = np.squeeze(stacked_windows)

stacked_labels_for_window = np.stack(stacked_labels_for_window, axis = 0)
stacked_labels_for_window = np.squeeze(stacked_labels_for_window)

stacked_window_times = np.stack(stacked_window_times, axis = 0)

# For each mini-spectrogram, find the average color across all unique syllables
mean_colors_per_minispec = np.zeros((stacked_labels_for_window.shape[0], 3))
for i in np.arange(stacked_labels_for_window.shape[0]):
    list_of_colors_for_row = [category_colors[x] for x in stacked_labels_for_window[i,:]]
    all_colors_in_minispec = np.array(list_of_colors_for_row)
    mean_color = np.mean(all_colors_in_minispec, axis = 0)
    mean_colors_per_minispec[i,:] = mean_color

# DO UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(stacked_windows)

plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], c = mean_colors_per_minispec)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Decomposition")
plt.show()

np.save(f'{folder_name}/embedding.npy', embedding)

# Now let's set up the data structure for the visualizer 

analysis_struct = {}
embStartEnd = np.zeros((2, stacked_window_times.shape[0]))

embStartEnd[0,:] = stacked_window_times[:,0]
embStartEnd[1,:] = stacked_window_times[:,-1]

analysis_struct['embVals'] = embedding.copy()
analysis_struct['behavioralArr'] = stacked_specs
analysis_struct['embStartEnd'] = embStartEnd

dat = np.savez(f'{folder_name}/analysis_dict.npz', **analysis_struct)



