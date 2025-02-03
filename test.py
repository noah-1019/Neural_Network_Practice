# Author: Noah Plant
# Date:   1/31/25
# Purpose:
# The purpose of this script is to create a NN that when given 11 inputs about a red wine it can determine the quality on a scale of
# 0 to 10. The data was found on: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download


# Import Libraries
import numpy as np 
import pandas as pd
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import keras
from nnTools import myfunctions as nn


#-------------------------------------#
# Create Functions                    #
#-------------------------------------#

def MSE (x,y):
    n=len(x)
    residual2=(x-y)**2
    total=sum(residual2)
    mse=total/n
    return(mse)

#-------------------------------------#
#  Read in dataframe and format data  #
#-------------------------------------#

df=pd.read_csv("winequality-red.csv")

# Split data into inputs and targets\
# inputs: 11
# targets: 1
targets=df['quality']
inputs=df.drop(columns='quality')

# Normalize data
# NN do better when the data is in the range of 0 to 1. 
# Here I just divided each row by the maximum in the 
# row to ensure the max value was and the min value is 0.
maxTarget=targets.max(axis=0)
maxInputs=inputs.max(axis=0)

targets=targets/maxTarget
inputs=inputs/maxInputs

# Split data into testing, validation, and training sections
# training 75%
# validation 15%
# testing 10%
numTrials=len(targets) #1599 trials
trainIndex=math.floor(.8*numTrials)
valIndex=math.floor(.95*numTrials)

# Create input dfs
trainInputs=inputs.iloc[0:trainIndex]
valInputs=inputs.iloc[trainIndex:valIndex]
testInputs=inputs.iloc[valIndex:numTrials+1]

# Create target dfs
trainTargets=targets.iloc[0:trainIndex]
valTargets=targets.iloc[trainIndex:valIndex]
testTargets=targets.iloc[valIndex:numTrials+1]

# This variable is the number of inputs that we will use to train the NN.
input_shape=[trainInputs.shape[1]]


#-------------------------------------#
# Create the Neural Network           #
#-------------------------------------#

# Find optimal neuron number #
# ----------------------------#

mse=[]
for i in range (2,40): # Iterates through the neuron number
    mseRow=[]
    for j in range(1): # Three trials for each neuron number


        # For the neural network I will create I will use a non linear activation funciton
        # This function is known as relu, this is a computationally efficient activation
        # function that ensures the gradient does not diverge.

        # Some questions concerning this section that I should review later:
        # What does making a Dense layer do?
        # What does the neuron number in the input layer do?

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=i,activation = 'relu',                              # Adds the input layer (10 neurons)
                                input_shape=input_shape),
            tf.keras.layers.Dense(units=i, activation='relu'),                              # Adds the hidden layer (10 neurons)
            tf.keras.layers.Dense(units=1)                                                   # Adds the output layer (1 neuron)

            
            ])

        # Some questions concerning this section that I should review later:
        # What does compiling do?
        # What is the adam optimizer
        # What is loss='mae'

        model.compile(optimizer='adam',loss='mae')
        
        model.summary()


        #-------------------------------------#
        # Train the Neural Network            #
        #-------------------------------------#

        # This stopping callback function allows the computer to determine the optimal epoch number
        # It will stop when the validation performance does worse than the training performance.

        # Some information on the paramaters:
        #-------------------------------------#

        # monitor: The value that is being monitored by the stopping function. Thre are two options
        # validation loss or validation accuracy, here I chose validation loss. If the validation
        # loss increases the stopping function stops the training.

        # min_delta: The minimum value should be set for the change to be considered, i.e. only 
        # changes above the min_delta will trigger any sort of reaction with the stopping condition.

        # patience: Patience is the number of epochs of training after the first halt. Basically the 
        # NN will train a little more after it has stopped to ensure that the stopping condition was 
        # not just a spike in validation loss.

        # verbose: This paramater determines what is outputed to the console during training. 
        # 0 = silent
        # 1 = progress bar
        # 2 = each epoch a line is printed to the screen

        # mode: Mode works with the monitor paramater, it should almost always be set to auto.
        # When the monitor is set to accuracy, the mode is "max", when the monitor is set to loss
        # the mode is set to "min."

        # baseline
        # restore_best_weights: This is a boolean value, a True value restores the weights which are optimal.

        earlystopping=keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=5, verbose=3,
                                                    mode="auto",baseline=None, restore_best_weights=False)




        losses=model.fit(trainInputs, trainTargets,
                        
                        validation_data=(valInputs,valTargets),
                        batch_size=256,# Usually this number is an exponent of 2
                        epochs=1000,
                        callbacks=[earlystopping])


        #-------------------------------------#
        # Predict on the data                 #
        #-------------------------------------#

        # Makes the prediction
        predictions=model.predict(testInputs)
        x=predictions.flatten().squeeze()






        

        # Calculate MSE
        y=testTargets.to_numpy()
        y=y.flatten().squeeze()

        


        mseRow.append(MSE(x,y))
    mse.append(mseRow)


numpyMSE=np.array(mse)

meanMse=numpyMSE.mean(axis=1)


#print(meanMse)

optimalNeurons=2+np.argmin(meanMse)

#print(optimalNeurons)

# Now that the optimal neuron number has been found, create another NN with 
# the best neuron number.



#-------------------------------------#
# Train Final Model                   #
#-------------------------------------#

model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=optimalNeurons,activation = 'relu',                              # Adds the input layer (10 neurons)
                                input_shape=input_shape),
            tf.keras.layers.Dense(units=optimalNeurons, activation='relu'),                              # Adds the hidden layer (10 neurons)
            tf.keras.layers.Dense(units=1)                                                   # Adds the output layer (1 neuron)

            
            ])

model.compile(optimizer='adam',loss='mae')
earlystopping=keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=5, verbose=3,
                                                    mode="auto",baseline=None, restore_best_weights=False)

losses=model.fit(trainInputs, trainTargets,
                        
                        validation_data=(valInputs,valTargets),
                        batch_size=256,# Usually this number is an exponent of 2
                        epochs=1000,
                        callbacks=[earlystopping])



# Visualize training progress
    # It is important to understand what is going on behind the hood of
    # the NN training. Plotting the validation, testing, and training
    # data is critical in developing an understanding of the 
    # situation.
loss_df=pd.DataFrame(losses.history)# history stores the loss/val for each epoch

loss_df.loc[:,['loss','val_loss']].plot()
plt.show()


# Makes the prediction
predictions=model.predict(testInputs)
x=predictions.flatten().squeeze()

# Calculate MSE
y=testTargets.to_numpy()
y=y.flatten().squeeze()
finalMse=MSE(x,y)

# Creates a regression line for the actual data
coef=np.polyfit(x,testTargets,1)
# # print(coef)
 
poly1d_fn=np.poly1d(coef) # regression line functions

plt.plot(x,testTargets,'yo',x,poly1d_fn(x),'--k')
plt.show()

r2=r2_score(y,x)
print(r2)

nn.plotRegression(x,y)
