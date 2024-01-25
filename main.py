#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:15:03 2024

@author: akapoor
"""

import numpy as np
import os
# absolute_path = '/home/akapoor'
absolute_path = '/Users/AnanyaKapoor'
os.chdir(f'{absolute_path}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Transection_Analysis/')
# sys.path.append('/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR/USA_5207_Analysis/')
from util import Canary_Analysis, DataPlotter
import umap
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# =============================================================================
#     # Set data parameters
# =============================================================================

# Let's create a list of source directories for which we will create Python files for

days_for_analysis = [14, 40]
source_dir_list = [f'{absolute_path}/Dropbox (University of Oregon)/USA5207/14', f'{absolute_path}/Dropbox (University of Oregon)/USA5207/40']
dest_dir_list = [f'{absolute_path}/Dropbox (University of Oregon)/AK_RHV_Analysis/USA5207/Python_Files/14', f'{absolute_path}/Dropbox (University of Oregon)/AK_RHV_Analysis/USA5207/Python_Files/40']

# Initialize the Canary_Analysis object

num_spec = [100, 7] # How many spectrogram from day at position 0, how many spectrogram from day at position 1
window_size = 100 # The window size for UMAP 
stride = 10 # Stride size for analysis 

# This is the upstream location where analysis results are stored.
analysis_path = f'{absolute_path}/Dropbox (University of Oregon)/AK_RHV_Analysis/USA5207/UMAP_Analysis/'

# Define the folder name. THe results will be directly stored here.
days_string = '_'.join(map(str, days_for_analysis))

num_spec_string = '_'.join(map(str, num_spec))
folder_name = f'{analysis_path}Days_For_Analysis_{days_string}_Num_Spectrograms_{num_spec_string}_Window_Size_{window_size}_Stride_{stride}' # 


lowThresh = 500
highThresh = 7000

masking_freq_tuple = (lowThresh, highThresh)
spec_dim_tuple = (window_size, 151) # dimensions of spec slices that are passed into UMAP 

transection_obj = Canary_Analysis(num_spec, window_size, stride, folder_name, masking_freq_tuple, spec_dim_tuple)
transection_obj.days_for_analysis = days_for_analysis
transection_obj.analysis_path = analysis_path
transection_obj.days_string = days_string
# Now I want to write code that will process each day

dest_dir_list = [f'{filepath}/songs' for filepath in dest_dir_list]

# Create Python Files from the raw wav files

transection_obj.organize_files(source_dir_list, dest_dir_list)

stacked_specs_list = []
stacked_labels_list = []
category_colors_list = []
stacked_windows_list = []
stacked_labels_for_window_list = []
stacked_window_times_list = []
mean_colors_per_minispec_list = []


for i in np.arange(len(days_for_analysis)):
    day = days_for_analysis[i]
    print(f'Processing Day {day} ...')
    stacked_specs, stacked_labels, category_colors, stacked_windows, stacked_labels_for_window, stacked_window_times, mean_colors_per_minispec, folder_name = transection_obj.process_day(dest_dir_list[i], day)    
    stacked_specs_list.append(stacked_specs)
    stacked_labels_list.append(stacked_labels)
    category_colors_list.append(category_colors)
    stacked_windows_list.append(stacked_windows)
    stacked_labels_for_window_list.append(stacked_labels_for_window)
    stacked_window_times_list.append(stacked_window_times)
    mean_colors_per_minispec_list.append(mean_colors_per_minispec)

stacked_windows = np.concatenate((stacked_windows_list))
mean_colors_per_minispec = np.concatenate((mean_colors_per_minispec_list))
stacked_window_times = np.concatenate((stacked_window_times_list), axis = 0)
# Need to z-score the stacked_windows, otherwise will be thrown an error message

mean = np.mean(stacked_windows, axis=1, keepdims=True)
std_dev = np.std(stacked_windows, axis=1, keepdims=True)

# Perform z-scoring
z_scored = (stacked_windows - mean) / std_dev

# Replace NaNs with 0s
z_scored = np.nan_to_num(z_scored)

# DO UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(z_scored)

plt.figure()

# I want to extract the indices of the stacked_windows that correspond to each day
day_labels = []

for i in np.arange(len(days_for_analysis)):
    day = days_for_analysis[i]
    day_windows = stacked_labels_for_window_list[i]
    individ_day = day*np.ones((stacked_labels_for_window_list[i].shape[0],1))
    day_labels.append(individ_day)
    
day_labels = np.concatenate((day_labels))
for d in days_for_analysis:
    indices = day_labels == d
    indices.shape = (indices.shape[0],)
    plt.scatter(embedding[indices,0], embedding[indices,1], c = mean_colors_per_minispec[indices, :], alpha = 0.8, label = f'Day {d}')
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Decomposition")
plt.legend()
plt.savefig(f'{folder_name}/UMAP_embedding.png')
plt.show()

np.save(f'{folder_name}/embedding.npy', embedding)

# =============================================================================
# Bokeh Plotting
# =============================================================================

list_of_images = []
stacked_windows_plotting = stacked_windows.copy()
stacked_windows_plotting.shape = (stacked_windows.shape[0], 1, 100, 151)


for i in np.arange(stacked_windows_plotting.shape[0]):
    data = stacked_windows_plotting[i,:,:,:]
    list_of_images.append(data)


embeddable_images = transection_obj.get_images(list_of_images)

transection_obj.plot_UMAP_embedding(embedding, mean_colors_per_minispec,embeddable_images, f'{folder_name}/UMAP_analysis.html', saveflag = True)


# Now let's set up the data structure for the visualizer 

analysis_struct = {}
embStartEnd = np.zeros((2, stacked_window_times.shape[0]))

embStartEnd[0,:] = stacked_window_times[:,0]
embStartEnd[1,:] = stacked_window_times[:,-1]

analysis_struct['embVals'] = embedding.copy()
analysis_struct['behavioralArr'] = stacked_specs
analysis_struct['embStartEnd'] = embStartEnd
analysis_struct['mean_colors_per_minispec'] = mean_colors_per_minispec

dat = np.savez(f'{folder_name}/analysis_dict.npz', **analysis_struct)

app = QApplication([])

# Instantiate the plotter    
plotter = DataPlotter()

# Accept folder of data
#plotter.accept_folder('SortedResults/B119-Jul28')
#/Users/ethanmuchnik/Desktop/Series_GUI/SortedResults/Pk146-Jul28/1B.npz
plotter.plot_file(f'{folder_name}/analysis_dict.npz', plotter)
# plotter.plot_file('/home/akapoor/Downloads/total_dict_hard_MEGA.npz')

# plotter.plot_file('working.npz')

plotter.addROI() # This plots the ROI circle

# plotter.zooming_in()

# Show
plotter.show()

