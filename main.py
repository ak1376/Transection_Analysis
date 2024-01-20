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
absolute_path = '/home/akapoor'
os.chdir(f'{absolute_path}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Transection_Analysis/')
# sys.path.append('/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR/USA_5207_Analysis/')
from util import WavtoSpec, Tweetyclr, DataPlotter
import pickle
import umap
import matplotlib.pyplot as plt
from PIL import Image
import base64
import io
from io import BytesIO
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, ColumnDataSource
import pandas as pd
import matplotlib.cm as cm
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import shutil

# TODO: Include capacity for colorizing by multiple days and loading data from multiple days. 

# =============================================================================
#     # Set data parameters
# =============================================================================

# Let's create a list of source directories for which we will create Python files for

days_for_analysis = [14, 40]
source_dir_list = [f'{absolute_path}/Dropbox (University of Oregon)/USA5207/14', f'{absolute_path}/Dropbox (University of Oregon)/USA5207/40']
dest_dir_list = [f'{absolute_path}/Dropbox (University of Oregon)/AK_RHV_Analysis/USA5207/Python_Files/14', f'{absolute_path}/Dropbox (University of Oregon)/AK_RHV_Analysis/USA5207/Python_Files/40']

# for i in np.arange(len(source_dir_list)):
#     source_dir = source_dir_list[i]
#     dest_dir = dest_dir_list[i]
#     os.makedirs(dest_dir, exist_ok=True)
#     spectrogram_creator = WavtoSpec(source_dir, dest_dir)
#     spectrogram_creator.process_directory()
    
#     # I want to create a helper function that groups the Python files into 
#     # songs and not songs. We will only be using songs for analysis   
    
#     # Extract the names of the song files
#     song_names = os.listdir(f'{source_dir}/corrected_song_mat_files')
    
#     updated_names = []
#     for name in song_names:
#         # Split the name at underscores and count them
#         parts = name.split('_')
#         if len(parts) > 2:
#             # Replace the second underscore with a dot
#             parts[1] = parts[1] + '.'
#             # Join the parts back together
#             modified_name = '_'.join(parts[:2]) + '_'.join(parts[2:])
#             # Remove the '.mat' extension
#             if modified_name.endswith('.mat'):
#                 modified_name = modified_name[:-4]
#             updated_names.append(modified_name)
#         else:
#             # If there are not enough underscores, just remove '.mat'
#             updated_names.append(name[:-4] if name.endswith('.mat') else name)
            
#     # Move the song files to another folder with the dest_dir
#     os.makedirs(f'{dest_dir}/songs', exist_ok=True)
#     song_dest_paths = [f'{dest_dir}/songs/{name}' + ".npz" for name in updated_names]
    
#     source_path_list = []
#     for file_path in updated_names:
#         source_path = f'{dest_dir}/{file_path}.npz'
#         dest_path = f'{dest_dir}/songs/{file_path}.npz'
#         # Move each file to the destination folder
#         shutil.move(source_path, dest_path)
    
#     # Now that we have moved all the songs to a folder, I will move the not 
#     # songs to another folder
#     os.makedirs(f'{dest_dir}/not_songs', exist_ok=True)
#     not_songs_source = [f'{dest_dir}/{file}' for file in os.listdir(dest_dir) if file.endswith('.npz')]
#     not_songs_dest = [f'{dest_dir}/not_songs/{file}' for file in os.listdir(dest_dir) if file.endswith('.npz')]
    
#     for i in np.arange(len(not_songs_source)):
#         source_path = not_songs_source[i]
#         dest_path = not_songs_dest[i]
        
#         shutil.move(source_path, dest_path)
        
# This is the upstream location where analysis results are stored.
analysis_path = f'{absolute_path}/Dropbox (University of Oregon)/AK_RHV_Analysis/USA5207/UMAP_Analysis/'

# Now let's define the location where we will store the results

num_spec = [100, 7] # How many spectrogram from day at position 0, how many spectrogram from day at position 1

# Parameters we set
window_size = 100 # The window size for UMAP 
stride = 10 # Stride size for analysis 

# Define the folder name. THe results will be directly stored here.
days_string = '_'.join(map(str, days_for_analysis))

num_spec_string = '_'.join(map(str, num_spec))
folder_name = f'{analysis_path}Days_For_Analysis_{days_string}_Num_Spectrograms_{num_spec_string}_Window_Size_{window_size}_Stride_{stride}' # 

lowThresh = 500
highThresh = 7000
# masking_freq_tuple = (lowThresh, highThresh) # Apply low and high pass filtering 
spec_dim_tuple = (window_size, 151) # dimensions of spec slices that are passed into UMAP 

# Now I want to write code that will process each day

dest_dir_list = [f'{filepath}/songs' for filepath in dest_dir_list]

def windowing(stacked_specs, stacked_labels, times, category_colors):
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

    
    return stacked_windows, stacked_labels_for_window, stacked_window_times, mean_colors_per_minispec


def process_day(dest_dir_path, day):
    bird_dir = dest_dir_path
    all_songs_data = [f'{bird_dir}/{element}' for element in os.listdir(bird_dir)] # Get the file paths of each numpy file from Yarden's data
    all_songs_data.sort() # Sort spectrograms chronologically 
    # I want to subset all_songs_data to only look at the specified number of spectrograms
    index = days_for_analysis.index(day)
    if len(all_songs_data)<num_spec[index]:
        num_spec[index] = len(all_songs_data)
    
    num_spec_string = '_'.join(map(str, num_spec))
    folder_name = f'{analysis_path}Days_For_Analysis_{days_string}_Num_Spectrograms_{num_spec_string}_Window_Size_{window_size}_Stride_{stride}' # 
    all_songs_data_subset = all_songs_data[0:num_spec[index]]
    
    os.makedirs(folder_name, exist_ok=True)
    num_spec_value = num_spec[index]

    # For each spectrogram we will extract
    # 1. Each timepoint's syllable label
    # 2. The spectrogram itself
    stacked_labels = [] 
    stacked_specs = []
    for i in np.arange(num_spec_value):
        # Extract the data within the numpy file. We will use this to create the spectrogram
        dat = np.load(all_songs_data_subset[i])
        spec = dat['s']
        times = dat['t']
        frequencies = dat['f']
        labels = dat['labels'] # Labels that will be used for color coding UMAP points
        labels = labels.T
        
        # I want to replace the labels with the day of analysis
        labels[:] = day



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
    
    # Now do the windowing procedure
    stacked_windows, stacked_labels_for_window, stacked_window_times, mean_colors_per_minispec = windowing(stacked_specs, stacked_labels, times, category_colors)
    
    return stacked_specs, stacked_labels, category_colors, stacked_windows, stacked_labels_for_window, stacked_window_times, mean_colors_per_minispec, folder_name


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
    stacked_specs, stacked_labels, category_colors, stacked_windows, stacked_labels_for_window, stacked_window_times, mean_colors_per_minispec, folder_name = process_day(dest_dir_list[i], day)    
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

def embeddable_image(data):
    data = (data.squeeze() * 255).astype(np.uint8)
    # convert to uint8
    data = np.uint8(data)
    image = Image.fromarray(data)
    image = image.rotate(90, expand=True) 
    image = image.convert('RGB')
    # show PIL image
    im_file = BytesIO()
    img_save = image.save(im_file, format='PNG')
    im_bytes = im_file.getvalue()

    img_str = "data:image/png;base64," + base64.b64encode(im_bytes).decode()
    return img_str


def get_images(list_of_images):
    return list(map(embeddable_image, list_of_images))


def plot_UMAP_embedding(embedding, mean_colors_per_minispec, image_paths, filepath_name, saveflag = False):

    # Specify an HTML file to save the Bokeh image to.
    # output_file(filename=f'{self.folder_name}Plots/{filename_val}.html')
    output_file(filename = f'{filepath_name}')

    # Convert the UMAP embedding to a Pandas Dataframe
    spec_df = pd.DataFrame(embedding, columns=('x', 'y'))


    # Create a ColumnDataSource from the data. This contains the UMAP embedding components and the mean colors per mini-spectrogram
    source = ColumnDataSource(data=dict(x = embedding[:,0], y = embedding[:,1], colors=mean_colors_per_minispec))


    # Create a figure and add a scatter plot
    p = figure(width=800, height=600, tools=('pan, box_zoom, hover, reset'))
    p.scatter(x='x', y='y', size = 7, color = 'colors', source=source)

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = """
        <div>
            <h3>@x, @y</h3>
            <div>
                <img
                    src="@image" height="100" alt="@image" width="100"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
        </div>
    """

    p.add_tools(HoverTool(tooltips="""
    """))
    
    # Set the image path for each data point
    source.data['image'] = image_paths
    # source.data['image'] = []
    # for i in np.arange(spec_df.shape[0]):
    #     source.data['image'].append(f'{self.folder_name}/Plots/Window_Plots/Window_{i}.png')


    save(p)
    show(p)


list_of_images = []
stacked_windows_plotting = stacked_windows.copy()
stacked_windows_plotting.shape = (stacked_windows.shape[0], 1, 100, 151)


for i in np.arange(stacked_windows_plotting.shape[0]):
    data = stacked_windows_plotting[i,:,:,:]
    list_of_images.append(data)


embeddable_images = get_images(list_of_images)

plot_UMAP_embedding(embedding, mean_colors_per_minispec,embeddable_images, f'{folder_name}/UMAP_analysis.html', saveflag = True)


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

