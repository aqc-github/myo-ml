# LIGHT GRADIENT BOOSTING CLASSIFIER MODEL VERSION 2

# library imports
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


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


## DATA PREPROCESSING
# load the data
log.info("Loading the data ...")
df = pd.read_csv('data/subject_010.csv')
log.info(f"Loaded data.")