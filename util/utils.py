#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:06:23 2023

@author: akapoor
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import umap
import torch
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, ColumnDataSource
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import base64
import io
from io import BytesIO
from tqdm import tqdm
from scipy.io import wavfile
import random
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from matplotlib import cm
from PyQt5.QtCore import Qt
import shutil
from pathlib import Path
from scipy.signal import windows, spectrogram, ellip, filtfilt
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from matplotlib import cm
from PyQt5.QtCore import Qt
import numpy as np
from PyQt5.QtCore import QPointF
import matplotlib.pyplot as plt
# -------------
from pyqtgraph import DateAxisItem, AxisItem, QtCore

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class Tweetyclr:
    def __init__(self, num_spec, window_size, stride, folder_name, all_songs_data, masking_freq_tuple, spec_dim_tuple, exclude_transitions = False, category_colors = None):
        '''The init function should define:
            1. directory for bird
            2. directory for python files
            3. analysis path
            4. folder name 


            Additional tasks
            1. create the folder name if it does not exist already

        '''
        # self.bird_dir = bird_dir
        # self.directory = directory
        self.num_spec = num_spec
        self.window_size = window_size
        self.stride = stride
        # self.analysis_path = analysis_path
        self.category_colors = category_colors
        self.folder_name = folder_name
        self.all_songs_data = all_songs_data
        self.masking_freq_tuple = masking_freq_tuple
        self.freq_dim = spec_dim_tuple[1]
        self.time_dim = spec_dim_tuple[0]
        self.exclude_transitions = exclude_transitions

        # Create the folder if it doesn't already exist
        if not os.path.exists(folder_name+"/Plots/Window_Plots"):
            os.makedirs(folder_name+"/Plots/Window_Plots")
            print(f'Folder "{folder_name}" created successfully.')
        else:
            print(f'Folder "{folder_name}" already exists.')

    def first_time_analysis(self):

        # For each spectrogram we will extract
        # 1. Each timepoint's syllable label
        # 2. The spectrogram itself
        stacked_labels = [] 
        stacked_specs = []
        for i in np.arange(self.num_spec):
            # Extract the data within the numpy file. We will use this to create the spectrogram
            dat = np.load(self.all_songs_data[i])
            spec = dat['s']
            times = dat['t']
            frequencies = dat['f']
            labels = dat['labels']
            labels = labels.T


            # Let's get rid of higher order frequencies
            mask = (frequencies<self.masking_freq_tuple[1])&(frequencies>self.masking_freq_tuple[0])
            masked_frequencies = frequencies[mask]

            subsetted_spec = spec[mask.reshape(mask.shape[0],),:]
            
            stacked_labels.append(labels)
            stacked_specs.append(subsetted_spec)

            
        stacked_specs = np.concatenate((stacked_specs), axis = 1)
        stacked_labels = np.concatenate((stacked_labels), axis = 0)
        stacked_labels.shape = (stacked_labels.shape[0],1)


        # Get a list of unique categories (syllable labels)
        unique_categories = np.unique(stacked_labels)
        if self.category_colors == None:
            self.category_colors = {category: np.random.rand(3,) for category in unique_categories}
            self.category_colors[0] = np.zeros((3)) # SIlence should be black
            # open a file for writing in binary mode
            with open(f'{self.folder_name}/category_colors.pkl', 'wb') as f:
                # write the dictionary to the file using pickle.dump()
                pickle.dump(self.category_colors, f)

        spec_for_analysis = stacked_specs.T
        window_labels_arr = []
        embedding_arr = []
        # Find the exact sampling frequency (the time in miliseconds between one pixel [timepoint] and another pixel)
        print(times.shape)
        dx = np.diff(times)[0,0]

        # We will now extract each mini-spectrogram from the full spectrogram
        stacked_windows = []
        # Find the syllable labels for each mini-spectrogram
        stacked_labels_for_window = []
        # Find the mini-spectrograms onset and ending times 
        stacked_window_times = []

        # The below for-loop will find each mini-spectrogram (window) and populate the empty lists we defined above.
        for i in range(0, spec_for_analysis.shape[0] - self.window_size + 1, self.stride):
            # Find the window
            window = spec_for_analysis[i:i + self.window_size, :]
            # Get the window onset and ending times
            window_times = dx*np.arange(i, i + self.window_size)
            # We will flatten the window to be a 1D vector
            window = window.reshape(1, window.shape[0]*window.shape[1])
            # Extract the syllable labels for the window
            labels_for_window = stacked_labels[i:i+self.window_size, :]
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
        # dict_of_spec_slices_with_slice_number = {i: stacked_windows[i, :] for i in range(stacked_windows.shape[0])}

        # For each mini-spectrogram, find the average color across all unique syllables
        mean_colors_per_minispec = np.zeros((stacked_labels_for_window.shape[0], 3))
        for i in np.arange(stacked_labels_for_window.shape[0]):
            list_of_colors_for_row = [self.category_colors[x] for x in stacked_labels_for_window[i,:]]
            all_colors_in_minispec = np.array(list_of_colors_for_row)
            mean_color = np.mean(all_colors_in_minispec, axis = 0)
            mean_colors_per_minispec[i,:] = mean_color

        self.stacked_windows = stacked_windows
        self.stacked_labels_for_window = stacked_labels_for_window
        self.mean_colors_per_minispec = mean_colors_per_minispec
        self.stacked_window_times = stacked_window_times
        self.masked_frequencies = masked_frequencies
        # self.dict_of_spec_slices_with_slice_number = dict_of_spec_slices_with_slice_number


    # def embeddable_image(self, data, folderpath_for_slices, window_times, iteration_number):
    #     # This function will save an image for each mini-spectrogram. This will be used for understanding the UMAP plot.
        
    #     window_data = data[iteration_number, :]
    #     window_times_subset = window_times[iteration_number, :]
    
    #     window_data.shape = (self.window_size, int(window_data.shape[0]/self.window_size))
    #     window_data = window_data.T 
    #     window_times = window_times_subset.reshape(1, window_times_subset.shape[0])
    #     plt.pcolormesh(window_times, self.masked_frequencies, window_data, cmap='jet')
    #     # let's save the plt colormesh as an image.
    #     plt.savefig(f'{folderpath_for_slices}/Window_{iteration_number}.png')
    #     plt.close()
    
    def embeddable_image(self, data):
        data = (data.squeeze() * 255).astype(np.uint8)
        # convert to uint8
        data = np.uint8(data)
        image = Image.fromarray(data)
        image = image.convert('RGB')
        # show PIL image
        im_file = BytesIO()
        img_save = image.save(im_file, format='PNG')
        im_bytes = im_file.getvalue()
    
        img_str = "data:image/png;base64," + base64.b64encode(im_bytes).decode()
        return img_str
    
    
    def get_images(self, list_of_images):
        return list(map(self.embeddable_image, list_of_images))


    def compute_UMAP_decomp(self, zscored):
        # Perform a UMAP embedding on the dataset of mini-spectrograms
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(zscored)

        return embedding

    def plot_UMAP_embedding(self, embedding, mean_colors_per_minispec, image_paths, filepath_name, saveflag = False):

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

    def find_slice_actual_labels(self, stacked_labels_for_window):
        al = []
        for i in np.arange(stacked_labels_for_window.shape[0]):
            arr = stacked_labels_for_window[i,:]
            unique_elements, counts = np.unique(arr, return_counts=True)
            # print(unique_elements)
            # print(counts)
            sorted_indices = np.argsort(-counts)
            val = unique_elements[sorted_indices[0]]
            if val == 0:
                if unique_elements.shape[0]>1:
                    val = unique_elements[sorted_indices[1]]
            al.append(val)

        actual_labels = np.array(al)
        
        self.actual_labels = actual_labels

    def shuffling(self, shuffled_indices = None):
        
        if shuffled_indices is None:
            shuffled_indices = np.random.permutation(self.stacked_windows.shape[0])
                    
        self.shuffled_indices = shuffled_indices
        
        
    def train_test_split(self, dataset, train_split_perc, shuffled_indices):
        ''' 
        The following is the procedure I want to do for the train_test_split.
        
        '''
        
        # I want to make training indices to be the first 80% of the shuffled data
        split_point = int(train_split_perc*dataset.shape[0])
        
        anchor_indices = shuffled_indices[:split_point]

        # Shuffle array1 using the shuffled indices
        stacked_windows_for_analysis_modeling = dataset[shuffled_indices,:]
        # Shuffle array2 using the same shuffled indices
        stacked_labels_for_analysis_modeling= self.stacked_labels_for_window[shuffled_indices,:]
        mean_colors_per_minispec_for_analysis_modeling = self.mean_colors_per_minispec[shuffled_indices, :]
        
        stacked_windows_train = torch.tensor(dataset[anchor_indices,:])
        stacked_windows_train = stacked_windows_train.reshape(stacked_windows_train.shape[0], 1, self.time_dim, self.freq_dim)
        anchor_indices = anchor_indices
        # self.train_indices = np.array(training_indices)
        
        mean_colors_per_minispec_train = self.mean_colors_per_minispec[anchor_indices,:]
        stacked_labels_train = self.stacked_labels_for_window[anchor_indices,:]
        
        
        anchor_train_indices = anchor_indices
        
        return stacked_windows_train, stacked_labels_train, mean_colors_per_minispec_train, anchor_indices 
        
        
class Temporal_Augmentation:
    
    def __init__(self, total_dict, simple_tweetyclr, tau_in_steps):
        self.total_dict = total_dict
        self.tweetyclr_obj = simple_tweetyclr
        self.tau = tau_in_steps
    
    def __call__(self, x):
        batch_data = x[0]
        indices = x[1]
        
        # Find the number of augmentations we want to use (up until tau step ahead)
        num_augs = np.arange(1, self.tau+1, 1).shape[0]
    
        # Preallocate tensors with the same shape as batch_data
        
        positive_aug_data = torch.empty(num_augs, 1, self.tweetyclr_obj.time_dim, self.tweetyclr_obj.freq_dim)
        
        # positive_aug_1 = torch.empty_like(batch_data)
        # positive_aug_2 = torch.empty_like(batch_data)
        
        total_training_indices = list(self.total_dict.keys())
        
        positive_aug_indices = indices + np.arange(1,self.tau+1, 1)      
            
        if any(elem in indices for elem in np.sort(total_training_indices)[-self.tau:]):
            positive_aug_indices = indices - np.arange(1,self.tau+1, 1)   
            
        try:
            # Your code that might raise the exception
            for i in np.arange(num_augs):
                positive_aug_data[i, :,:,:] = torch.tensor(self.total_dict[int(positive_aug_indices[i])].reshape(batch_data.shape[0], 1, self.tweetyclr_obj.time_dim, self.tweetyclr_obj.freq_dim))
        
        except ValueError as e:
            print(f"Encountered KeyError: {e}. Press Enter to continue...")
            input()
            

        return positive_aug_data

class Custom_Contrastive_Dataset(Dataset):
    def __init__(self, tensor_data, slice_indices, tensor_labels, transform=None):
        self.data = tensor_data
        self.slice_indices = slice_indices
        self.labels = tensor_labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        index = self.slice_indices[index]
        lab = self.labels[index]
        
        x = [x, lab, index]
        x1 = self.transform(x) if self.transform else x

        return [x1, index, lab]


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        # Get the two augmentations from jawn
        aug = self.transform(x)
        return [aug[i, :, :, :] for i in range(aug.shape[0])]
    
class WavtoSpec:
    def __init__(self, src_dir, dst_dir):
        self.src_dir = src_dir
        self.dst_dir = dst_dir

    def process_directory(self):
        # First walk to count all the .wav files
        total_files = sum(
            len([f for f in files if f.lower().endswith('.wav')])
            for r, d, files in os.walk(self.src_dir)
        )
        
        # Now process each file with a single tqdm bar
        with tqdm(total=total_files, desc="Overall progress") as pbar:
            for root, dirs, files in os.walk(self.src_dir):
                dirs[:] = [d for d in dirs if d not in ['.DS_Store']]  # Ignore irrelevant directories
                files = [f for f in files if f.lower().endswith('.wav')]
                for file in files:
                    full_path = os.path.join(root, file)
                    self.convert_to_spectrogram(full_path)
                    pbar.update(1)  # Update the progress bar for each file
    
    def convert_to_spectrogram(self, file_path, min_length_ms=1000):
        try:
            # Read the WAV file
            samplerate, data = wavfile.read(file_path)

            # Calculate the length of the audio file in milliseconds
            length_in_ms = (data.shape[0] / samplerate) * 1000

            if length_in_ms < min_length_ms:
                print(f"File {file_path} is below the length threshold and will be skipped.")
                return  # Skip processing this file

            # High-pass filter (adjust the filtering frequency as necessary)
            b, a = ellip(5, 0.2, 40, 500/(samplerate/2), 'high')
            data = filtfilt(b, a, data)

            # Canary song analysis parameters
            NFFT = 1024  # Number of points in FFT
            step_size = 119  # Step size for overlap

            # Calculate the overlap in samples
            overlap_samples = NFFT - step_size

            # Use a Gaussian window
            window = windows.gaussian(NFFT, std=NFFT/8)

            # Compute the spectrogram with the Gaussian window
            f, t, Sxx = spectrogram(data, fs=samplerate, window=window, nperseg=NFFT, noverlap=overlap_samples)
            f.shape = (f.shape[0],1)
            t.shape = (1, t.shape[0])

            # Convert to dB
            Sxx_log = 10 * np.log10(Sxx)

            # Post-processing: Clipping and Normalization
            clipping_level = -2  # dB
            Sxx_log_clipped = np.clip(Sxx_log, a_min=clipping_level, a_max=None)
            Sxx_log_normalized = (Sxx_log_clipped - np.min(Sxx_log_clipped)) / (np.max(Sxx_log_clipped) - np.min(Sxx_log_clipped))

            # Visualization (optional)
            # plt.imshow(Sxx_log_normalized, aspect='auto', origin='lower', extent=[t.min(), t.max(), f.min(), f.max()])
            # plt.title('Spectrogram')
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [sec]')
            # plt.colorbar(format='%+2.0f dB')
            # plt.show()

            # Assuming label is an integer or float
            labels = np.full((Sxx_log_normalized.shape[1],), 0)  # Adjust the label array as needed
            labels.shape = (1, labels.shape[0])

            # Define the path where the spectrogram will be saved
            spec_filename = os.path.splitext(os.path.basename(file_path))[0]
            spec_file_path = os.path.join(self.dst_dir, spec_filename + '.npz')

            # Saving the spectrogram and the labels
            np.savez_compressed(spec_file_path, s=Sxx_log_normalized, t = t, f = f, labels=labels)

            # Print out the path to the saved file
            print(f"Spectrogram saved to {spec_file_path}")

        except ValueError as e:
            print(f"Error reading {file_path}: {e}")


    def visualize_random_spectrogram(self):
        # Get a list of all '.npz' files in the destination directory
        npz_files = list(Path(self.dst_dir).glob('*.npz'))
        if not npz_files:
            print("No spectrograms available to visualize.")
            return
        
        # Choose a random spectrogram file
        random_spec_path = random.choice(npz_files)
        
        # Load the spectrogram data from the randomly chosen file
        with np.load(random_spec_path) as data:
            spectrogram_data = data['s']
        
        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram_data, aspect='auto', origin='lower')
        plt.title(f"Random Spectrogram: {random_spec_path.stem}")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(format='%+2.0f dB')
        plt.show()

    # Helper function to find the next power of two
    def nextpow2(x):
        return np.ceil(np.log2(np.abs(x))).astype('int')
    
    def copy_yarden_data(src_dirs, dst_dir):
        """
        Copies all .npz files from a list of source directories to a destination directory.
    
        Parameters:
        src_dirs (list): A list of source directories to search for .npz files.
        dst_dir (str): The destination directory where .npz files will be copied.
        """
        # Ensure the destination directory exists
        Path(dst_dir).mkdir(parents=True, exist_ok=True)
    
        # Create a list to store all the .npz files found
        npz_files = []
    
        # Find all .npz files in source directories
        for src_dir in src_dirs:
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.endswith('.npz'):
                        npz_files.append((os.path.join(root, file), file))
    
        # Copy the .npz files to the destination directory with progress bar
        for src_file_path, file in tqdm(npz_files, desc='Copying files'):
            dst_file_path = os.path.join(dst_dir, file)
            
            # Ensure we don't overwrite files in the destination
            if os.path.exists(dst_file_path):
                print(f"File {file} already exists in destination. Skipping copy.")
                continue
    
            # Copy the .npz file to the destination directory
            shutil.copy2(src_file_path, dst_file_path)
            print(f"Copied {file} to {dst_dir}")
            
class DataPlotter(QWidget):
  
    def __init__(self):
        QWidget.__init__(self) # TODO? parent needed? #awkward


        # Setup main window/layout
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.app = pg.mkQApp()


        # Instantiate window
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('Embedding Analysis')

        # Behave plot
        self.behavePlot = self.win.addPlot()


        # Define bottom plot 
        self.win.nextRow()
        self.embPlot = self.win.addPlot()


        self.setupPlot()
        
        self.win.scene().sigMouseClicked.connect(self.update2)    

        self.additionList = []
        self.additionCount = 0.05 # An indicator value for the times in the spectrogram that correspond to each selected UMAP point. 


    def setupPlot(self):

        # Setup behave plot for img
        self.imgBehave = pg.ImageItem() # object that will store our spectrogram representation 
        self.behavePlot.addItem(self.imgBehave) # Populating the behavioral part of the plot with the spectrogram representation object
        self.behavePlot.hideAxis('left') # Reduces clutter
        self.behavePlot.setMouseEnabled(x=True, y=False)  # Enable x-axis and disable y-axis interaction

    def clear_plots(self):
        self.embPlot.clear()
        self.behavePlot.clear()


    def set_behavioral_image(self,image_array,colors_per_timepoint, **kwargs):
        '''
        This function should be able to plot without colors_per_timepoint in the situation where we do not have labeled data. 
        '''
        
        self.behave_array = image_array # image_array is our spectrogram representation 

        filterSpec = image_array
        # Normalize the numeric array to the [0, 1] range
        normalized_array = (filterSpec - np.min(filterSpec)) / (np.max(filterSpec) - np.min(filterSpec))

        # Apply the colormap to the normalized array

        rgb_array = plt.cm.get_cmap('inferno')(normalized_array)
        rgb_add = np.zeros_like(image_array) # array of zeros that is a placeholder for the total behavioral spectrogram array
        
# =============================================================================
#         ## If we have access to ground truth labels for each timepoint then we will colorize by the timepoint.
# =============================================================================
        
        if colors_per_timepoint is not None:
            # I want to colorize the spectrogram by timepoint.
            colors = np.concatenate((colors_per_timepoint, np.ones((colors_per_timepoint.shape[0],1))), axis = 1)

            if 'addition' in kwargs.keys(): # If we have selected at least one ROI
                relList = kwargs['addition']
                rgb_add1 = colors.copy() 
                reshaped_colors = np.tile(rgb_add1, (rgb_array.shape[0], 1, 1)) # Assign the same RGB color to each of the frequency pixels (pixels along the y-axis)
                rgb_add1 = reshaped_colors.reshape(rgb_array.shape[0], rgb_array.shape[1], rgb_array.shape[2])
                for img in relList: # For all the points in the ROI
                    rgb_add += img # populate the empty rgb_add
                zero_columns = np.all(rgb_add == 0, axis=0)
                zero_columns_indices = np.where(zero_columns)[0]
    
                rgb_add1[:,zero_columns_indices,:] = 0
    
            else:
                rgb_add1 = np.zeros_like(colors)
                
        else:
            if 'addition' in kwargs.keys():
                relList = kwargs['addition']
                for img in relList:
                    rgb_add += img 
            
            rgb_add1 = plt.cm.get_cmap('hsv')(rgb_add)
            rgb_add1[rgb_add == 0] = 0
            
        self.imgBehave.setImage(rgb_array + rgb_add1)
        
    def update(self):
        rgn = self.region.getRegion()
    
        findIndices = np.where(np.logical_and(self.startEndTimes[0,:] > rgn[0], self.startEndTimes[1,:] < rgn[1]))[0]
    
        self.newScatter.setData(pos = self.emb[findIndices,:])
    
        self.embPlot.setXRange(np.min(self.emb[:,0]) - 1, np.max(self.emb[:,0] + 1), padding=0)
        self.embPlot.setYRange(np.min(self.emb[:,1]) - 1, np.max(self.emb[:,1] + 1), padding=0)
        
    def prompt_timepoints(self):
        text, ok = QInputDialog.getText(self, 'Timepoint Input', 'Enter start and end timepoints (comma separated):')
        if ok and text:
            start, end = map(float, text.split(','))
            return start, end
        return None, None

    # Load the embedding and start times into scatter plot
    def accept_embedding(self,embedding,startEndTimes, mean_colors_for_minispec):

        self.emb = embedding
        self.startEndTimes = startEndTimes

        if mean_colors_for_minispec is not None:
            colors = np.concatenate((mean_colors_for_minispec, np.ones((mean_colors_for_minispec.shape[0],1))), axis = 1)
            colors*=255
        else:
            self.cmap = cm.get_cmap('hsv')
            norm_times = np.arange(self.emb.shape[0])/self.emb.shape[0]
            colors = self.cmap(norm_times) * 255

        # colors = self.cmap(norm_times) * 255
        self.defaultColors = colors.copy()
        self.scatter = pg.ScatterPlotItem(pos=embedding, size=5, brush=colors)
        self.embPlot.addItem(self.scatter)
        
        # Below two lines are necessary for plotting the highlighted points. 
        self.newScatter = pg.ScatterPlotItem(pos=embedding[0:10,:], size=10, brush=pg.mkBrush(255, 255, 255, 200))
        self.embPlot.addItem(self.newScatter) 


        # Scale imgBehave 
        height,width = self.behave_array.shape

        x_start, x_end, y_start, y_end = 0, self.startEndTimes[1,-1], 0, height
        pos = [x_start, y_start]
        scale = [float(x_end - x_start) / width, float(y_end - y_start) / height]

        self.imgBehave.setPos(*pos)
        tr = QtGui.QTransform() #Transformation object. Allows us to interact with the graph and have the graph change. 
        self.imgBehave.setTransform(tr.scale(scale[0], scale[1])) # Interact with particular scaling factors 
        
        # x_start, x_end = self.prompt_timepoints()
        
        self.behavePlot.getViewBox().setLimits(yMin=y_start, yMax=y_end)
        self.behavePlot.getViewBox().setLimits(xMin=x_start, xMax=x_end)

        # print(self.startEndTimes)
        self.region = pg.LinearRegionItem(values=(0, self.startEndTimes[0,-1] / 10))
        self.region.setZValue(10)

        
        self.region.sigRegionChanged.connect(self.update)

        self.behavePlot.addItem(self.region)


        # consider where    

        self.embMaxX = np.max(self.emb[:,0])
        self.embMaxY = np.max(self.emb[:,1])


        self.embMinX = np.min(self.emb[:,0])
        self.embMinY = np.min(self.emb[:,1])

        self.embPlot.setXRange(self.embMinX - 1, self.embMaxX + 1, padding=0)
        self.embPlot.setYRange(self.embMinY - 1, self.embMaxY + 1, padding=0)
        
    def zooming_in(self):
        height,width = self.behave_array.shape
        x_start, x_end = self.prompt_timepoints()
        y_start, y_end = 0, height
        
        self.behavePlot.getViewBox().setLimits(yMin=y_start, yMax=y_end)
        self.behavePlot.getViewBox().setLimits(xMin=x_start, xMax=x_end)

    def plot_file(self,filePath):

        self.clear_plots()
        self.setupPlot()

        A = np.load(filePath)

        self.startEndTimes = A['embStartEnd']
        if 'colors_per_timepoint' in A:
            self.colors_per_timepoint = A['colors_per_timepoint']
        else:
            self.colors_per_timepoint = None
            
        self.behavioralArr = A['behavioralArr']
        plotter.set_behavioral_image(A['behavioralArr'], self.colors_per_timepoint)
        
        if 'mean_colors_per_minispec' in A:
            self.mean_colors_per_minispec = A['mean_colors_per_minispec']
        else:
            self.mean_colors_per_minispec = None

        # feed it (N by 2) embedding and length N list of times associated with each point
        plotter.accept_embedding(A['embVals'],A['embStartEnd'], self.mean_colors_per_minispec)
        
        
    def clear_previous_highlights(self):
        # Clear previous highlights logic
        self.additionList = self.additionList[-1:]
        self.additionCount = 0.05
        self.set_behavioral_image(self.behave_array,self.colors_per_timepoint, addition = self.additionList)
        # Any other necessary steps to clear the highlights
        
    def addROI(self):
        # # Ask user if they want to clear previous highlights
        # reply = QMessageBox.question(self, 'Clear Highlights', 
        #                              "Do you want to clear all previous highlights?", 
        #                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    
        # if reply == QMessageBox.Yes:
        #     self.clear_previous_highlights()
    
        # Rest of the method
        self.r1 = pg.EllipseROI([0, 0], [self.embMaxX/5, self.embMaxY/5], pen=(3,9))
        self.embPlot.addItem(self.r1)
        
    # def addROI(self):
    #     self.r1 = pg.EllipseROI([0, 0], [self.embMaxX/5, self.embMaxY/5], pen=(3,9))
    #     # r2a = pg.PolyLineROI([[0,0], [0,self.embMaxY/5], [self.embMaxX/5,self.embMaxY/5], [self.embMaxX/5,0]], closed=True)
    #     self.embPlot.addItem(self.r1)

        #self.r1.sigRegionChanged.connect(self.update2)

    # Manage key press events
    def keyPressEvent(self,evt):
        print('key is ',evt.key())

        if evt.key() == 65: # stick with numbers for now
            self.update()

    def update2(self):
        print('called')
        ellipse_size = self.r1.size()
        ellipse_center = self.r1.pos() + ellipse_size/2

        # try:
        #     self.outCircles = np.vstack((self.outCircles,np.array([ellipse_center[0],ellipse_center[1],ellipse_size[0],ellipse_size[1]])))
        # except:
        #     self.outCircles = np.array([ellipse_center[0],ellipse_center[1],ellipse_size[0],ellipse_size[1]])

        # # print(self.outCircles)
        # np.savez('bounds.npz',bounds = self.outCircles)
        # Print the center and size
        # print("Ellipse Center:", ellipse_center)
        # print("Ellipse Size:", ellipse_size)
        # print(self.r1)
        # print(ellipse_size[0])

        #manual distnace
        bound = np.square(self.emb[:,0] - ellipse_center[0])/np.square(ellipse_size[0]/2) +  np.square(self.emb[:,1] - ellipse_center[1])/np.square(ellipse_size[1]/2)
        indices_in_roi = np.where(bound < 1)[0]
        # print(f'The number of indices in the ROI is {indices_in_roi.shape}')
        print(indices_in_roi)
            
            
        # clear_highlights = input("Clear previous highlights? (y/n): ")
        # if clear_highlights == "y":
        #     self.additionList = []

        # # points_in_roi = [QPointF(x, y) for x, y in self.emb if self.r1.contains(QPointF(x, y))]
        # # print(points_in_roi)
        # print('does it contian 0,0')
        # if self.r1.contains(QPointF(0,0)):
        #     print('yes')

        # # indices_in_roi = [pt for pt in self.emb if roiShape.contains(pt)]
        # # print(roiShape.pos())
        # indices_in_roi = [index for index, (x, y) in enumerate(self.emb) if self.r1.contains(QPointF(x, y))]
        # print(indices_in_roi)
        # # print(indices_in_roi)
        # # print(self.emb.shape)

        tempImg = self.behave_array.copy()*0
        presumedTime = np.linspace(self.startEndTimes[0,0],self.startEndTimes[1,-1],num = tempImg.shape[1])
        # print(f'The Presumed Time: {presumedTime}')
        # print(self.startEndTimes.shape)
        # print(self.behave_array.shape)


        for index in indices_in_roi:
            # For each index in the ROI, extract the associated spec slice
            mask = (presumedTime < self.startEndTimes[1,index]) & (presumedTime > self.startEndTimes[0,index]) 
            print("MASK SHAPE")
            print(mask.shape)
            relPlace = np.where(mask)[0]
            print("RELPLACE")
            print(relPlace)
            print(relPlace.shape)
            tempImg[:,relPlace] = self.additionCount
            # print("WHAT IS ADDITION COUNT")
            # print(self.additionCount)

        self.additionList.append(tempImg)
        # print(f'The Shape of the Temporary Image: {tempImg.shape}')
        # print(f'The Length of the Addition List: {len(self.additionList)}')
        self.additionCount += .05
        
        
        if len(self.additionList) > 1:
            # Ask user if they want to clear previous highlights
            reply = QMessageBox.question(self, 'Clear Highlights', 
                                          "Do you want to clear all previous highlights?", 
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
            if reply == QMessageBox.Yes:
                self.clear_previous_highlights()

        self.set_behavioral_image(self.behave_array,self.colors_per_timepoint, addition = self.additionList)
        

        
        # self.newScatter.setData(pos = self.emb[indices_in_roi,:])

        # self.embPlot.setXRange(np.min(self.emb[:,0]) - 1, np.max(self.emb[:,0] + 1), padding=0)
        # self.embPlot.setYRange(np.min(self.emb[:,1]) - 1, np.max(self.emb[:,1] + 1), padding=0)


    def show(self):
        self.win.show()
        self.app.exec_()
        

