# **EDSA Apple Prices Challenge - Regression Analysis**

Regression_AM1 - Innocentia Kati, Nonkululeko Ntuli, Thanyani, Boitumelo Makgoba, Kago
**The structure of this notebook is as follows:**

[**1 Introduction**](##1-introduction)

>[2.1 Import modules](#21-import-modules)

>[2.2 Import dataset](#22-import-dataset)

>[2.3 EDA](#23-eda)

>[2.4 Modelling](#24-modelling)

>[2.5 Model selection](#25-model-selection)

>[2.6 Prediction](#Prediction)

## 1. Introduction

In a Fresh Produce Industry, How much stock do you have on hand? Not too little that you run out of stock when customers want to buy more. And not too much that food waste occurs. How do you set your prices? Yields from farms fluctuate by season. Should your prices then also fluctuate by season?

The aim of this project is to construct a regression algorithm, capable of accurately predicting how much a kilogram of Golden Delicious Apples will cost, given certain parameters.

For the predictions, the train-set data will be used.

The variable to be predicted(y) is Average Price per Kg

## 2. Body


### 2.1 Import modules

  from sklearn.linear_model import LinearRegression
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, VotingRegressor
  from catboost import CatBoostRegressor
  from sklearn.model_selection import GridSearchCV
  from sklearn.neighbors import KNeighborsRegressor

  from sklearn.model_selection import train_test_split
  from sklearn.model_selection import cross_val_score

  from sklearn.preprocessing import StandardScaler

  from sklearn.metrics import r2_score
  from sklearn.metrics import mean_squared_error

  #%matplotlib notebook
  import missingno
  import seaborn as sns
  import matplotlib.pyplot as plt
  %matplotlib inline
  import plotly.express as px

  from scipy import stats
  import math
  import pickle
  import numpy as np
  import pandas as pd


### 2.2 Dataset

##### Description of files in datasets:

  * Train - Dataset training our model
  train = pd.read_csv('df_train_set.csv')
  
  * Test - Dataset testing our model (y variables to be predicted)
  test = pd.read_csv('df_test_set.csv')
  
  NB: We used jupyter notebook to run codes to import. The data was read from one file.

### 2.3 EDA
Categorical Variables: 
* Province
* Container
* Size Grade
* Commodities
* Date

Total number of Variables:
13 (including the target variable)

  
#### List of graphs created to explore data included in our notebook:

  - Figure 1. Overall data distribution: distribution of each of the feature in the 'train' dataset
  - Figure 2. Density plot of each feature
  - Figure 3. Bar plot, Size Grade
  - Figure 4. Checking correlations: correlations heat map for the variables in the 'train' dataset.
  - figure 5. distribution of the target variable
  
  ### 2.4 Modelling
  
  #### 2.4.1 Preprocessing
  
  We started by analyzing the data and finding the columns that had any missing values, or had no relevance to the final prediction.
  
  
  #### 2.4.5 Models used
  
  Linear
  Decision Tree Regressor
  Random Forest, Gradient Boosting, Bagging, AdaBoost Regressors
  Cat Boost Regressors

### 2.5 Model selection

The moment we’ve all been waiting for!

We trained above different regression models and discovered that VotingRegressor returned the highest score, with a Train RMSE of 0.097 Test RMSE of 0.094.
