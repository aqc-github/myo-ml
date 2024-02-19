# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import zscore

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
log.info("Starting the dataset study ...")


# load the data 
df = pd.read_csv('data/subject_010_500ms.csv')
log.info(f"Loaded data.")

# remove the ts column
df.drop(['ts'], axis=1, inplace=True)
# this label holds too much information, remove it

target = 'grasp'
# turn the label into a categorical variable
df[target] = df[target].astype('category')  # Convert to Pandas categorical type

# EDA
# check the target variable
log.info(f"Data y: {df[target]}")
log.info(f"Data y unique: {df[target].unique()}") # this is the number of classes
log.info(f"Data y nunique: {df[target].nunique()}") # this is the number of classes
log.info(f"Data y value counts: {df[target].value_counts()}") # this is the number of classes

# check the features
features = df.columns.to_list()
features.remove(target)
log.info(f"Data X: {df[features]}") # all features
log.info(f"Data X shape: {df[features].shape}") # number of rows and columns
log.info(f"Data X columns: {df[features].columns}") # column names
log.info(f"Data X types: {df[features].dtypes}") # data types
log.info(f"Data X head: {df[features].head()}") # first 5 rows
log.info(f"Data X describe: {df[features].describe()}") # summary statistics
log.info(f"Data X info: {df[features].info()}") # number of non-null values for each feature
log.info(f"Data X isnull: {df[features].isnull().sum()}") # number of null values for each feature
log.info(f"Data X nunique: {df[features].nunique()}") # number of unique values for each feature
log.info(f"Data X value counts: {df[features].value_counts()}") # value counts for each feature
log.info(f"Data X corr: {df[features].corr()}") # correlation matrix
# the correlation matrix seems to be 349x349 which is too big to plot, find the figures with correlation bigger than 0.8, 0.7, 0.6
corr = df[features].corr()
corr_08 = corr[corr > 0.8]
corr_07 = corr[corr > 0.7]
corr_06 = corr[corr > 0.6]
# plot the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr_08)
plt.show()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_07)
plt.show()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_06)
plt.show()

# eliminate features with high correlation, correlation > 0.8
# df.drop(['feature1', 'feature2'], axis=1, inplace=True)



log.info(f"Data X skew: {df[features].skew()}") # skew > 0 means right-skewed, < 0 means left-skewed
log.info(f"Data X kurtosis: {df[features].kurtosis()}") # kurtosis > 0 means heavy tails, < 0 means light tails


# Handling missing values
imputer = SimpleImputer(strategy='mean')  # or median, mode
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# normalizing accounting for skewness
df_filled[features] = np.log1p(df_filled[features])


# Outlier Detection and Removal
# z_scores = np.abs(zscore(df_filled))
# df_no_outliers = df_filled[(z_scores < 3).all(axis=1)]

# Feature Scaling
scaler = StandardScaler()
features_to_scale = df_filled.columns.difference([target])
df_scaled = df_filled.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df_filled[features_to_scale])


# Class Imbalance
# SMOTE
smote = SMOTE(random_state=42)
# 
X, y = smote.fit_resample(df_scaled[features], df_scaled[target])
df_smote = pd.concat([X, y], axis=1)

# Feature Selection
# Variance Threshold
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(df_smote[features])
# plot the variance with a demanding threshold at 0.7
# variance is best when it is close to 1
plt.figure(figsize=(10, 6))
plt.plot(selector.variances_)
plt.axhline(y=0.7, color='r', linestyle='-')
plt.xlabel('Features')
plt.ylabel('Variance')
plt.title('Variance Threshold')
plt.show()
# log results
log.info(f"Data X shape (variance): {df_smote[features].shape}")
log.info(f"Data X selected shape (variance): {X_selected.shape}")
log.info(f"Data X columns (variance): {df_smote[features].columns}")
log.info(f"Data X selected columns (variance): {df_smote[features].columns[selector.get_support()]}")


# Assuming 'X' is your feature set and 'y' is your target variable:
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

# Transform X to the selected features
X_best = selector.transform(X)

# Now you can access the p-values with the following attribute (assuming your target variable is categorical):
p_values = -np.log10(selector.pvalues_)

# Then you can plot these p-values:
plt.figure(figsize=(10, 6))
plt.plot(p_values)
plt.title("P-Values of Features")
plt.show()

# Log results
log.info(f"Data X shape (p-values): {X.shape}")
log.info(f"Data X best shape (p-values): {X_best.shape}")
log.info(f"Data X columns (p-values): {X.columns}")
log.info(f"Data X best columns (p-values): {X.columns[selector.get_support()]}")
log.info(f"Data X best p-values (p-values): {p_values}")


# Principal Component Analysis
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_best)  # Use fit_transform here

# plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')  # Representing cumulative variance explained by components
plt.title('Explained Variance')
plt.show()

# Log results
log.info(f"Data X shape (pca): {X_best.shape}")
log.info(f"Data X pca shape (pca): {X_pca.shape}")  # Now X_pca has the correct shape
log.info(f"Data X columns (pca): {df_smote[features].columns}")
# When logging PCA columns, there is no direct correspondence with original features, as PCA creates new components
log.info(f"Number of PCA components: {pca.n_components_}")
log.info(f"Data X pca explained variance ratio (pca): {pca.explained_variance_ratio_}")





