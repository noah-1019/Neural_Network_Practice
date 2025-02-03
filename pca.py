# Author: Noah Plant
# Date:   2/1/25
# Purpose:
# The purpose of this script is to reduce the dimensionality of a data set using PCA, random tree, and an autoencoder,
# While this data set is pretty small and has no need to be reduced, it is a nice simple data set that is easy to
# practice on. The data set can be found here: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download

# Import Libraries
import numpy as np 
import pandas as pd
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import keras
import statistics

## -------------------------------------------------------------------------------- ##
# Load in and format data                                                            #
## -------------------------------------------------------------------------------- ##

df=pd.read_csv("winequality-red.csv")

# Split data into inputs and targets\
# inputs: 11
# targets: 1
targets=df['quality']
inputs=df.drop(columns='quality')

# Normalize data
# PCA does better when the data is in the range of 0 to 1. 
# Here I just divided each row by the maximum in the 
# row to ensure the max value was and the min value is 0.
maxTarget=targets.max(axis=0)
maxInputs=inputs.max(axis=0)

targets=targets/maxTarget
inputs=inputs/maxInputs

# Split data into testing, validation, and training sections
# training 90%
# testing 10%
numTrials=len(targets) #1599 trials
trainIndex=math.floor(.9*numTrials)


# Create input dfs
trainInputs=inputs.iloc[0:trainIndex]
testInputs=inputs.iloc[trainIndex:numTrials+1]

# Create target dfs
targets_train=targets.iloc[0:trainIndex]
targets_test=targets.iloc[trainIndex:numTrials+1]


## -------------------------------------------------------------------------------- ##
# Apply PCA funciton                                                                 #
## -------------------------------------------------------------------------------- ##

pca=PCA(n_components=11) # Splits the data into 11 components.

inputs_train=pca.fit_transform(trainInputs)
inputs_test=pca.fit_transform(testInputs)

explained_variance=pca.explained_variance_ratio_

# Displays the principle component values
for indx, evr in enumerate(explained_variance):
    print(f"PC{indx+1}: {evr:.2f}")

names=("PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11")

width = 0.8  # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
bottom = np.zeros(11)

p = ax.bar(names, explained_variance, width, label=names, bottom=bottom)


ax.set_title('PC Importance')

plt.show()

## -------------------------------------------------------------------------------- ##
# Auto Encoder Practice                                                              #
## -------------------------------------------------------------------------------- ##

# Like PCA Autoencoders reduce the dimensionality of a data set allowing for faster more
# efficient NN training. Auto encoders are especially useful when construction deep NN 
# used to analyze images.