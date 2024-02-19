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
path = "data/"

# mat files for all subjects from s010 to s115 
# mat_files = ["S010_ex1.mat", "S011_ex1.mat", "S012_ex1.mat", "S013_ex1.mat", "S014_ex1.mat", 
#              "S015_ex1.mat", "S016_ex1.mat", "S017_ex1.mat", "S018_ex1.mat", "S019_ex1.mat", 
#           "S020_ex1.mat", "S021_ex1.mat", "S022_ex1.mat", "S023_ex1.mat", "S024_ex1.mat", 
 #            "S025_ex1.mat", "S026_ex1.mat", "S027_ex1.mat", "S028_ex1.mat", "S029_ex1.mat", 
  #           "S030_ex1.mat", "S031_ex1.mat", "S032_ex1.mat", "S033_ex1.mat", "S034_ex1.mat", 
   #          "S035_ex1.mat", "S036_ex1.mat", "S037_ex1.mat", "S038_ex1.mat", "S039_ex1.mat", 
    #         "S040_ex1.mat", "S101_ex1.mat", "S102_ex1.mat", "S103_ex1.mat", "S104_ex1.mat", 
     #        "S105_ex1.mat", "S106_ex1.mat", "S107_ex1.mat", "S108_ex1.mat", "S109_ex1.mat",
      #       "S110_ex1.mat", "S111_ex1.mat", "S112_ex1.mat", "S113_ex1.mat", "S114_ex1.mat","S115_ex1.mat"]
# mat_files = ["S108_ex1.mat", "S109_ex1.mat",
#             "S110_ex1.mat", "S111_ex1.mat", "S112_ex1.mat", "S113_ex1.mat", "S114_ex1.mat","S115_ex1.mat"]
             
mat_files = ["S010_ex1.mat"] # for testing purposes

# initialize variables
time_window = 2000 # 100 ms
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

# Function to classify the grasp label for a window of data, prioritizing active states.
def classify_grasp_label(window):
    """
    Classify the grasp label for a window of data, prioritizing active states.
    
    Args:
    window (array-like): A window of data containing grasp labels.
    
    Returns:
    int: The classified grasp label.
    """
    # Convert the window to a list of labels
    labels = list(window)

    # Filter out the neutral state (assuming it's represented by 0)
    active_labels = [label for label in labels if label != 0]

    if active_labels:
        # If there are active labels, prioritize them
        # Option 1: Return the most frequent active label
        most_frequent_label = max(set(active_labels), key=active_labels.count)
        return most_frequent_label

        # Option 2: Return the active label with the highest value
        # return max(active_labels)
    else:
        # If there are no active labels, return the neutral state
        return 0



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
        features[f"MAV{i}"] = np.mean(np.abs(window[:, i]))
        # Zero Crossing (ZC)
        features[f"ZC{i}"] = np.sum(np.abs(np.diff(np.sign(window[:, i])))) / (2 * window.shape[0])
        # Slope Sign Change (SSC)
        features[f"SSC{i}"] = np.sum(np.diff(np.sign(np.diff(window[:, i]))) != 0) / window.shape[0]
        # Waveform Length (WL)
        features[f"WL{i}"] = np.sum(np.abs(np.diff(window[:, i]))) / window.shape[0]
        # Root Mean Square (RMS)
        features[f"RMS{i}"] = np.sqrt(np.mean(np.square(window[:, i])))
        # Variance (VAR)
        features[f"VAR{i}"] = np.var(window[:, i])
        # Log Detector (LOG)
        epsilon = 1e-10
        features[f"LOG{i}"] = np.exp(np.mean(np.log(np.abs(window[:, i]) + epsilon)))
        # Integrated EMG (IEMG)
        features[f"IEMG{i}"] = np.sum(np.abs(window[:, i]))

        # Frequency domain features (assuming FFT is applicable)
        # The actual frequencies will depend on the sampling rate and window length
        fft_vals = np.fft.rfft(window[:, i])
        fft_freqs = np.fft.rfftfreq(window.shape[0])

        # Mean Frequency (MNF)
        features[f"MNF_real{i}"] = np.sum(np.abs(fft_vals.real) * fft_freqs) / np.sum(np.abs(fft_vals.real))
        features[f"MNF_imag{i}"] = np.sum(np.abs(fft_vals.imag) * fft_freqs) / np.sum(np.abs(fft_vals.imag))
        # Median Frequency (MDF)
        features[f"MDF_real{i}"] = np.median(np.abs(fft_vals.real) * fft_freqs) / np.median(np.abs(fft_vals.real))
        features[f"MDF_imag{i}"] = np.median(np.abs(fft_vals.imag) * fft_freqs) / np.median(np.abs(fft_vals.imag))
        # Mean Power Frequency (MPF)
        features[f"MPF_real{i}"] = np.sum(np.square(np.abs(fft_vals.real)) * fft_freqs) / np.sum(np.square(np.abs(fft_vals.real)))
        features[f"MPF_imag{i}"] = np.sum(np.square(np.abs(fft_vals.imag)) * fft_freqs) / np.sum(np.square(np.abs(fft_vals.imag)))
        # Note: Peak Frequency (PF) depends on the magnitude and not directly on fft_vals, so it remains unchanged.
        features[f"PF{i}"] = fft_freqs[np.argmax(np.abs(fft_vals))]
        # Frequency Variance (FV) - Here, it's important to note that MNF is separated into real and imaginary. 
        # You might want to recompute it inside this block to maintain clarity.
        features[f"FV_real{i}"] = np.sum(np.square(np.abs(fft_vals.real) - features[f"MNF_real{i}"]) * fft_freqs) / np.sum(np.square(np.abs(fft_vals.real)))
        features[f"FV_imag{i}"] = np.sum(np.square(np.abs(fft_vals.imag) - features[f"MNF_imag{i}"]) * fft_freqs) / np.sum(np.square(np.abs(fft_vals.imag)))
        # Max and Min frequencies remain unchanged as they are based on magnitudes.
        features[f"MAXF{i}"] = fft_freqs[np.argmax(np.abs(fft_vals))]
        features[f"MINF{i}"] = fft_freqs[np.argmin(np.abs(fft_vals))]
        # Standard Deviation Frequency (SDF)
        features[f"SDF_real{i}"] = np.std(fft_vals.real)
        features[f"SDF_imag{i}"] = np.std(fft_vals.imag)
        # Skewness and Kurtosis Frequency
        features[f"SKF_real{i}"] = np.sum(np.power(np.abs(fft_vals.real) - features[f"MNF_real{i}"], 3) * fft_freqs) / np.sum(np.power(np.abs(fft_vals.real) - features[f"MNF_real{i}"], 3))
        features[f"SKF_imag{i}"] = np.sum(np.power(np.abs(fft_vals.imag) - features[f"MNF_imag{i}"], 3) * fft_freqs) / np.sum(np.power(np.abs(fft_vals.imag) - features[f"MNF_imag{i}"], 3))
        features[f"KUF_real{i}"] = np.sum(np.power(np.abs(fft_vals.real) - features[f"MNF_real{i}"], 4) * fft_freqs) / np.sum(np.power(np.abs(fft_vals.real) - features[f"MNF_real{i}"], 4))
        features[f"KUF_imag{i}"] = np.sum(np.power(np.abs(fft_vals.imag) - features[f"MNF_imag{i}"], 4) * fft_freqs) / np.sum(np.power(np.abs(fft_vals.imag) - features[f"MNF_imag{i}"], 4))
        # Maximum Power Spectral Density (MAXPSD)
        features[f"MAXPSD{i}_real"] = np.max(np.square(fft_vals.real))
        features[f"MAXPSD{i}_imag"] = np.max(np.square(fft_vals.imag))

        # Minimum Power Spectral Density (MINPSD)
        features[f"MINPSD{i}_real"] = np.min(np.square(fft_vals.real))
        features[f"MINPSD{i}_imag"] = np.min(np.square(fft_vals.imag))

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
        features["ts"] = np.mean(window[:, 12])
        # add an additional feature to the dataframe: the grasp label of the window
        features["grasp"] = classify_grasp_label(window[:, -1])

        # log both features for debugging, ts and grasp label
        # log.debug(f"Features: {features}")
        # log.debug(f"ts: {features['ts']}")
        # log.debug(f"grasp: {features['grasp']}")
        
        # Append the features of this window to the accumulated DataFrame
        allfeatures = pd.concat([allfeatures, features], axis=0, ignore_index=True)

        # log.debug feature dimensions for debugging
        # log.debug(f"Features shape: {allfeatures.shape}")



    return allfeatures

    

# Function to save a subjects data once extracted and processed
# into a .csv file
def save_subject_data(subject_data, subject_id):
    # save data into .csv file
    subject_data.to_csv(f"data/subject_{subject_id}_2000ms.csv", index=False)
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


