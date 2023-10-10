import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os 
import glob

from rich.logging import RichHandler
from rich.progress import track
import logging

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")
log.info("Starting the program ...")


# Load the data
path = "data/processed_data/"
path_single = "data/processed_data/subject_010.csv"

# Load the data
def load_and_combine_data(path):
    # Load all the csv files from the path
    os.chdir(path)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    log.info(f"Loaded {len(all_filenames)} files from {path}.")

    def load_data(filepath):
        df = pd.read_csv(filepath)
        log.info(f"Loaded {filepath}.")
        return df

    combined_csv = pd.concat([load_data(f) for f in track(all_filenames, description="Loading CSV files...")])
    log.info(f"Combined {len(all_filenames)} files into one dataframe.")
    
    return combined_csv


# Load the data forma single csv file
def load_single_csv(path):
    df = pd.read_csv(path)
    log.info(f"Loaded {path}.")
    return df




from sklearn.model_selection import KFold

# function to train XGBoost model
def train_xgboost(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model

# function to train Random Forest model
def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model

# function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# function to plot accuracy
def plot_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=['Accuracy'], y=[accuracy])
    plt.title('Accuracy')
    plt.show()

# function to plot mean squared error
def plot_mse(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=['MSE'], y=[mse])
    plt.title('Mean Squared Error')
    plt.show()

# function to plot actual vs predicted
def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 7))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted')
    plt.show()

# function to plot feature importance
def plot_feature_importance(model, features):
    plt.figure(figsize=(10, 7))
    sns.barplot(x=model.feature_importances_, y=features.columns)
    plt.title('Feature Importance')
    plt.show()


def k_fold_xg(df, k=10):
    # Split the data into features and target
    X = df.drop(['grasp', 'ts'], axis=1)
    y = df['grasp']

    # Define the k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Initialize variables to keep track of the best model
    best_accuracy = 0
    best_model = None
    best_model_type = ''
    best_X_test = None
    best_y_test = None

    # Loop through all folds
    for fold, (train_index, test_index) in track(enumerate(kf.split(X, y)), description="Running k-fold cross-validation..."):
        log.info(f"Starting fold {fold + 1}/{k}...")

        # Split data into train and test sets for the current fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        log.info(f"Split data into train and test sets for fold {fold + 1}/{k}.")

        # Train and evaluate XGBoost model
        y_pred_xgb, xg_model = train_xgboost(X_train, X_test, y_train, y_test)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        log.info(f"Trained XGBoost model for fold {fold + 1}/{k}.")

        # Train and evaluate Random Forest model
        y_pred_rf, rf_model = train_random_forest(X_train, X_test, y_train, y_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        log.info(f"Trained Random Forest model for fold {fold + 1}/{k}.")

        # Check which model performed better and save it if it's the best so far
        if accuracy_xgb > accuracy_rf and accuracy_xgb > best_accuracy:
            best_accuracy = accuracy_xgb
            best_model = xg_model
            best_model_type = 'XGBoost'
            best_X_test = X_test
            best_y_test = y_test
            log.info(f"Best model is XGBoost with accuracy {best_accuracy}. In fold {fold + 1}/{k}.")
        elif accuracy_rf > accuracy_xgb and accuracy_rf > best_accuracy:
            best_accuracy = accuracy_rf
            best_model = rf_model
            best_model_type = 'Random Forest'
            best_X_test = X_test
            best_y_test = y_test
            log.info(f"Best model is Random Forest with accuracy {best_accuracy}. In fold {fold + 1}/{k}.")

        log.info(f"Completed fold {fold + 1}/{k}.")

    # After all folds, plot the best model's results
    log.info(f"Best model is {best_model_type} with accuracy {best_accuracy}.")
    best_y_pred = best_model.predict(best_X_test)
    plot_confusion_matrix(best_y_test, best_y_pred)
    plot_accuracy(best_y_test, best_y_pred)
    plot_mse(best_y_test, best_y_pred)
    plot_actual_vs_predicted(best_y_test, best_y_pred)
    plot_feature_importance(best_model, X.dropna(axis=1))

    log.info(f"Finished k-fold cross-validation.")


# ---------------------------------------------------------------------------------------------
# 2, PCA to reduce the number of features
# function to perform PCA
def perform_pca(df, n_components=10):
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(df)
    log.info(f"Performed PCA with {n_components} components.")
    return pca











# ---------------------------------------------------------------------------------------------
# MAIN:

# Load the data
df = load_single_csv(path_single)
log.info(df.head())

# Print only the grasp and ts columns
log.info(df[['grasp', 'ts']].head())

# To run k-fold cross-validation
k_fold_xg(df, k=10)