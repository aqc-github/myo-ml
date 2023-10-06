import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# setup logging and rich progress bar
import logging
import time
from rich.logging import RichHandler
from rich.progress import track, Progress


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")
log.info("Starting the program ...")


# INTIALIZE VARIABLES #
# vector with all .mat files
path = "data/NINAPRO/MP1/"
mat_files = ["S106_ex1.mat","S107_ex1.mat","S108_ex1.mat","S109_ex1.mat","S110_ex1.mat","S111_ex1.mat"
             ,"S112_ex1.mat","S113_ex1.mat","S114_ex1.mat","S115_ex1.mat"] 

#mat_files = ["S010_ex1.mat"] # for testing purposes

# initialize variables
time_window = 100 # 100 ms
sample_rate = 1926 # 1926 Hz
window_size: int = time_window * sample_rate / 1000 # 192.6 samples (round to 192)
window_overlap = 0.2 # 20% overlap, 38.4 samples (round to 38)
num_windows = int((window_size - window_overlap) / (window_size - window_overlap)) + 1 # 5 windows per second



## ------------------------------------------------------------- ##
## FUNCTION DEFINITIONS ##

# Function to extract data windows from a single .mat file
# extrcat just emg, ts and grasp labels
def extract_data(mat_file):
    # load .mat file
    mat = scipy.io.loadmat(mat_file)
    # extract 12 emg channels from .mat file
    emg = mat["emg"]
    # extract ts from .mat file
    ts = mat["ts"]
    # extract grasp abels from .mat file
    grasp = mat["grasp"]

    
    return emg, ts, grasp



# Function to extract features from a single data window
# The following features are extracted:
# - Mean Absolute Value (MAV)
# - Zero Crossing (ZC)
# - Slope Sign Change (SSC)
# - Waveform Length (WL)
# - Root Mean Square (RMS)
# - Variance (VAR)
# - Log Detector (LOG)
# - Integrated EMG (IEMG)
# - Mean Frequency (MNF)
# - Median Frequency (MDF)
# - Mean Power Frequency (MPF)
# - Peak Frequency (PF)
# - Frequency Variance (FV)
# - Maximum Frequency (MAXF)
# - Minimum Frequency (MINF)
# - Standard Deviation Frequency (SDF)
# - Skewness Frequency (SKF)
# - Kurtosis Frequency (KUF)
# - Maximum Power Spectral Density (MAXPSD)
# - Minimum Power Spectral Density (MINPSD)
# All parameters will be calculated for all 12 channels in the domains of time and frequency
def extract_features_from_window(window):
    # Initialize feature dictionary
    features = {}

    # Calculate features for each of the channels
    for i in range(12):


        # Time domain features
        # Mean Absolute Value (MAV)
        features[f"MAV{i}"] = np.mean(np.abs(window[i, :]))
        # Zero Crossing (ZC)
        features[f"ZC{i}"] = np.sum(np.abs(np.diff(np.sign(window[i, :])))) / (2 * window.shape[1])
        # Slope Sign Change (SSC)
        features[f"SSC{i}"] = np.sum(np.diff(np.sign(np.diff(window[i, :])) != 0)) / window.shape[1]
        # Waveform Length (WL)
        features[f"WL{i}"] = np.sum(np.abs(np.diff(window[i, :]))) / window.shape[1]
        # Root Mean Square (RMS)
        features[f"RMS{i}"] = np.sqrt(np.mean(np.square(window[i, :])))
        # Variance (VAR)
        features[f"VAR{i}"] = np.var(window[i, :])
        # Log Detector (LOG)
        epsilon = 1e-10  # A small constant to avoid log(0)
        features[f"LOG{i}"] = np.exp(np.mean(np.log(np.abs(window[i, :]) + epsilon)))
        # Integrated EMG (IEMG)
        features[f"IEMG{i}"] = np.sum(np.abs(window[i, :]))


        # Frequency domain features
        # Mean Frequency (MNF)
        features[f"MNF{i}"] = np.sum(np.abs(window[i, :]) * np.arange(window.shape[1])) / np.sum(np.abs(window[i, :]))
        # Median Frequency (MDF)
        features[f"MDF{i}"] = np.median(np.abs(window[i, :]) * np.arange(window.shape[1])) / np.median(np.abs(window[i, :]))
        # Mean Power Frequency (MPF)
        features[f"MPF{i}"] = np.sum(np.square(np.abs(window[i, :])) * np.arange(window.shape[1])) / np.sum(np.square(np.abs(window[i, :])))
        # Peak Frequency (PF)
        if np.any(window[i, :]):
            features[f"PF{i}"] = np.argmax(np.abs(window[i, :])) / window.shape[1]
        else:
            features[f"PF{i}"] = 0  # Assign a default value or handle it according to your requirements
        # Frequency Variance (FV)
        features[f"FV{i}"] = np.sum(np.square(np.abs(window[i, :]) - features[f"MNF{i}"]) * np.arange(window.shape[1])) / np.sum(np.square(np.abs(window[i, :])))
        # Maximum Frequency (MAXF) - Handle empty sequences
        if np.any(window[i, :]):
            features[f"MAXF{i}"] = np.max(np.abs(window[i, :])) / window.shape[1]
        else:
            features[f"MAXF{i}"] = 0  # Assign a default value or handle it according to your requirements
        # Minimum Frequency (MINF)
        if np.any(window[i, :]):
            features[f"MINF{i}"] = np.min(np.abs(window[i, :])) / window.shape[1]
        else:
            features[f"MINF{i}"] = 0  # Assign a default value or handle it according to your requirements
        # Standard Deviation Frequency (SDF)
        features[f"SDF{i}"] = np.std(np.abs(window[i, :])) / window.shape[1]
        # Skewness Frequency (SKF)
        features[f"SKF{i}"] = np.sum(np.power(np.abs(window[i, :]) - features[f"MNF{i}"], 3) * np.arange(window.shape[1])) / np.sum(np.power(np.abs(window[i, :]) - features[f"MNF{i}"], 3))
        # Kurtosis Frequency (KUF)
        features[f"KUF{i}"] = np.sum(np.power(np.abs(window[i, :]) - features[f"MNF{i}"], 4) * np.arange(window.shape[1])) / np.sum(np.power(np.abs(window[i, :]) - features[f"MNF{i}"], 4))
        # Maximum Power Spectral Density (MAXPSD)
        features[f"MAXPSD{i}"] = np.max(np.square(np.abs(window[i, :]))) / window.shape[1]
        # Minimum Power Spectral Density (MINPSD)
        features[f"MINPSD{i}"] = np.min(np.square(np.abs(window[i, :]))) / window.shape[1]

    # transform dictionary into pandas dataframe keeping the order of the keys and names
    # of the columns
    features = pd.DataFrame([features])

    return features
    



# Function to extract data windows from a single .mat file
# perform feature extraction on each emg channels in those windows 
# write the data in an auxiliary .csv file
def extract_features(mat_file, window_size, window_overlap):

    # extract .mat file
    emg, ts, grasp = extract_data(mat_file)
    # merge emg, ts and grasp into one array
    data = np.concatenate((emg, ts, grasp), axis=1)
    # extract number of channels
    num_channels = emg.shape[1]
    # extract number of samples
    num_samples = emg.shape[0]
    # extract number of windows
    num_windows = int((num_samples - window_size) / (window_size - window_overlap)) + 1

    # define all features
    allfeatures = pd.DataFrame()


    # extract data windows
    # for i in track(range(num_windows), description=f'Extracting features from Subject ...'):
    for i in track(range(num_windows), description=f'Extracting features from Subject ...'):
        # extract data window
        window = data[int(i * (window_size - window_overlap)) : int(i * (window_size - window_overlap) + window_size), :]
        # extract features
        features = extract_features_from_window(window)
        # add an additional feature to the dataframe: the timestamp of the window
        features["ts"] = window[12, 0]
        # add an additional feature to the dataframe: the grasp label of the window
        features["grasp"] = window[12, 1]

        # Append the features of this window to the accumulated DataFrame
        allfeatures = pd.concat([allfeatures, features], axis=0, ignore_index=True)

        # log.debug feature dimensions for debugging
        # log.debug(f"Features shape: {allfeatures.shape}")



    return allfeatures

    

# Function to save a subjects data once extracted and processed
# into a .csv file
def save_subject_data(subject_data, subject_id):
    # save data into .csv file
    subject_data.to_csv(f"data/processed_data/subject_{subject_id}.csv", index=False)
    # log progress
    log.info(f"Data from subject {subject_id} saved", extra={"markup": True})




## ------------------------------------------------------------- ##
## MAIN ##
# extract features from all .mat files


# Data Extraction
log.info(f"Extracting data from [bold yellow]{mat_files}[/bold yellow]", extra={"markup": True})

# for loop to extract data from all .mat files, proccess it and extract the subject id
# save the data into a .csv file
for mat_file in mat_files:
    # extract features from .mat file
    features = extract_features(path + mat_file, window_size, window_overlap)
    # extract subject id from .mat file name
    subject_id = mat_file[1:4]
    # save subject data into .csv file
    save_subject_data(features, subject_id)
    

            
log.info(f"Data extraction complete for all subjects", extra={"markup": True})  


