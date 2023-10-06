import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from rich.logging import RichHandler
from rich.progress import track

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")
log.info("Starting the program ...")


# Load the data
path = "data/processed_data/"
# script to enter a folder and load all .csv files inside it
import os
import glob
os.chdir(path)
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
log.info(f"Loaded {len(all_filenames)} files from {path}.")
# combine all files in the list, with a tag for each file depending on the subject number
def load_data(path):
    df = pd.read_csv(path)
    return df
    log.info(f"Loaded {path}.")
# combine all files in the list
combined_csv = pd.concat([load_data(f) for f in all_filenames ])
log.info(f"Combined {len(all_filenames)} files into one dataframe.")


# Split the data into features and labels
# last key is the label: grasp
# second to last key is the time label for each window

# features
X = combined_csv.iloc[:, :-2].values
# labels
y = combined_csv.iloc[:, -1].values
# time labels
t = combined_csv.iloc[:, -2].values


from sklearn.model_selection import KFold

# Define the number of folds for cross-validation
n_splits = 10  # You can adjust this number as needed

# Initialize a cross-validation splitter
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

# Lists to store your training and testing data and labels for each fold
X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

# Split your data into training and testing sets for each fold
for train_index, test_index in kf.split(X):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    
    X_train_list.append(X_train_fold)
    X_test_list.append(X_test_fold)
    y_train_list.append(y_train_fold)
    y_test_list.append(y_test_fold)
