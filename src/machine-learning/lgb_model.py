# LIGHT GBM CLASSIFIER MODEL

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



# define functions for light gbm training and k-fold cross validation
# first, we testbench with the default parameters
# then, we use grid search to find the best parameters
# finally, we use the best parameters to train the model and make predictions

def train_lgb(X_train, y_train, X_test, y_test, params, verbose_eval=100):
    """
    Train light gbm model with given parameters
    """
    # create dataset for light gbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test)

    # train model
    model = lgb.train(params,
                      lgb_train,
                      num_boost_round=10000,
                      valid_sets=[lgb_train, lgb_test],
                      early_stopping_rounds=100,
                      verbose_eval=verbose_eval)

    return model

# function to round to the closest integer the 'grasp' column, and transform to int
def round_grasp(df):
    """
    Round grasp column
    """
    df['grasp'] = df['grasp'].round().astype(int)
    log.info(f"Rounded grasp column.")
    return df

# funcion to change integer column into grasp labels column: if 0, no grasp, if 1, medium wrap, etc.
label_names = {0: 'Neutral', 1:'medium_wrap', 2: 'lateral', 3: 'parallel_grasp', 4: 'tripod_grasp', 5: 'power_sphere', 6: 'precision_disk', 
               7: 'prismatic_pinch', 8: 'index_extension', 9: 'adducted_thumb', 10: 'prismatic_4_finger'}

def add_labels(df, target):
    """
    Add labels to grasp column
    """
    df[target] = df[target].apply(lambda x: label_names[x])
    log.info(f"Added labels to grasp column.")
    return df


# function to remove outliers from all columns except the target column
def remove_outliers(df, target):
    """
    Remove outliers from all columns except the target column.
    """
    for column in df.columns:
        if column != target:
            # Calculate mean and standard deviation for the column
            data_mean, data_std = np.mean(df[column]), np.std(df[column])

            # Define outliers as those more than 3 standard deviations from the mean
            cut_off = data_std * 3
            lower, upper = data_mean - cut_off, data_mean + cut_off

            # Remove rows where the column value is an outlier
            df = df[(df[column] >= lower) & (df[column] <= upper)]

    log.info("Removed outliers.")
    return df


# remove rows with nans
def remove_nans(df):
    """
    Remove NaNs
    """
    df = df.dropna()
    log.info(f"Removed NaNs.")
    return df

def normalize_data(df, target):
    """
    Normalize all columns except target.
    """
    for column in df.columns:
        if column != target:
            std = df[column].std()
            if std != 0:
                df[column] = (df[column] - df[column].mean()) / std
            else:
                # If standard deviation is zero, set the column to 0 or handle as needed
                df[column] = 0
    log.info("Normalized data.")
    return df


# split dataset into X and y
def split_dataset(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    log.info(f"Split dataset into X and y.")
    return X, y

# split dataset into train and test, 20% test size
def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    log.info(f"Split dataset into train and test.")
    return X_train, X_test, y_train, y_test


# define k-fold cross validation
def k_fold_cv(X, y, params, n_splits=5):
    """
    Perform k-fold cross validation
    """
    # create dataset for light gbm
    lgb_train = lgb.Dataset(X, y)

    # perform k-fold cross validation
    cv_results = lgb.cv(params,
                        lgb_train,
                        num_boost_round=10000,
                        nfold=n_splits,
                        early_stopping_rounds=100,
                        verbose_eval=10,
                        stratified=True)

    return cv_results

# define grid search
def grid_search(X, y, params, grid_params, n_splits=5):
    """
    Perform grid search
    """
    # create dataset for light gbm
    lgb_train = lgb.Dataset(X, y)

    # perform grid search
    gbm = lgb.LGBMClassifier()
    grid_search = GridSearchCV(estimator=gbm,
                               param_grid=grid_params,
                               cv=n_splits,
                               verbose=10,
                               n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search

# define function to make predictions
def predict(model, X_test):
    """
    Make predictions
    """
    # predict class probabilities
    # raw score : the prediction result of the model without sigmoid transformation
    # pred_leaf : whether to predict with leaf index, leaf index is the leaf number of a tree,
    #             its value is [0, num_leaf_left), for example, if we have a tree with 12 leaves,
    #             then the leaf index will be in [0, 12), so the leaf index of each sample will be an integer in [0, 12).
    y_pred = model.predict(X_test, num_iteration=model.best_iteration, raw_score = False, pred_leaf = False)
    
    # .predict returns probabilities
    # transform to class labels
    class_labels = np.argmax(y_pred, axis=1)


    return class_labels

# Define function to calculate classification metrics
def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics like accuracy, log loss, and confusion matrix
    """
    accuracy = accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred, labels=np.unique(y_true))
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, logloss, conf_matrix

# Define function to plot confusion matrix
def plot_confusion_matrix(conf_matrix):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Functions to plot classification metrics
def plot_classification_metrics(accuracy, logloss):
    """
    Plot classification metrics
    """
    plt.figure(figsize=(10, 7))
    plt.plot(accuracy, label='Accuracy')
    plt.plot(logloss, label='Log Loss')
    plt.legend()
    plt.show()

# Calculate ROC curve
def calculate_roc_curve(y_true, y_pred):
    """
    Calculate ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return fpr, tpr, thresholds

# Plot ROC curve
def plot_roc_curve(fpr, tpr):
    """
    Plot ROC curve
    """
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()





# main function
def main():
    # load data
    df = pd.read_csv('data/subject_010.csv')
    log.info(f"Loaded data.")

    # Before rounding
    print("Unique values before rounding:", df['grasp'].unique())

    # Apply rounding
    # NO LONGER NEED TO ROUND SINCE ALL LABELS ALREADY COME WITH ACTIVE STATE, BUT HELPS TURNING TO INT
    df = round_grasp(df)

    # After rounding
    print("Unique values after rounding:", df['grasp'].unique())

    # define target
    target = 'grasp'

    # remove outliers
    df = remove_outliers(df, target)

    nan_columns = df.columns[df.isna().any()].tolist()
    print(nan_columns)

    # remove NaNs
    df = remove_nans(df)

    nan_columns = df.columns[df.isna().any()].tolist()
    print(nan_columns)

    # normalize all columns except target
    df = normalize_data(df, target)

    nan_columns = df.columns[df.isna().any()].tolist()
    print(nan_columns)


    # split dataset into X and y
    X, y = split_dataset(df, target)

    nan_columns = X.columns[X.isna().any()].tolist()
    print(nan_columns)


    # split dataset into train and test
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # check for unique labels in y_train
    print("Unique values after rounding:", y_train.unique())

    # define parameters
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 11,
        'metric': 'multi_error',
        'num_leaves': 50,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'n_jobs': -1
    }

    # Define class weights
    # class weights are used to balance the dataset
    class_weights = {0: 1, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10}

    # Update params with class weights
    params['class_weight'] = class_weights

    # print head of X_train and y_train, to check if everything is ok
    log.info(f"X_train head: \n{X_train.head()}")
    log.info(f"y_train head: \n{y_train.head()}")



    # perform k-fold cross validation
    cv_results = k_fold_cv(X, y, params)
    log.info(f"CV results: {cv_results}")

    # perform grid search
    grid_params = {
        'learning_rate': [0.05, 0.1],
        'n_estimators': [8, 16, 24],
        'num_leaves': [6, 8, 12, 16],
        'boosting_type': ['gbdt', 'rf'],
        'objective': ['multiclass'],
        'num_class': [11],
        'random_state': [501],  # Updated from 'seed'
        'colsample_bytree': [0.65, 0.66],
        'subsample': [0.7, 0.75],
        'reg_alpha': [1, 1.2],
        'reg_lambda': [1, 1.2, 1.4],
    }

    grid_s = grid_search(X, y, params, grid_params)
    log.info(f"Grid search results: {grid_s}")

    # train model with best parameters
    best_params = grid_s.best_params_
    log.info(f"Best parameters: {best_params}")
    model = train_lgb(X_train, y_train, X_test, y_test, best_params)

    # make predictions
    y_pred = predict(model, X_test)
    log.info(f"Predictions: {y_pred}")
    

    # Now you can use predicted_labels with accuracy_score
    # empty y_true for now
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate classification metrics instead of RMSE
    accuracy, logloss, conf_matrix = calculate_classification_metrics(y_test, y_pred)
    log.info(f"Accuracy: {accuracy}, Log Loss: {logloss}")
    log.info(f"Confusion Matrix: \n{conf_matrix}")

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)

    # plot classification metrics
    plot_classification_metrics(accuracy, logloss)

    # calculate ROC curve
    fpr, tpr, thresholds = calculate_roc_curve(y_test, y_true)
    log.info(f"False Positive Rate: {fpr}, True Positive Rate: {tpr}, Thresholds: {thresholds}")

    # plot ROC curve
    plot_roc_curve(fpr, tpr)

    



if __name__ == "__main__":
    main()
