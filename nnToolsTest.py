
from nnTools import myfunctions as nn
import numpy as np
import pandas as pd

x=np.array([1,2,3,4,5])
y=np.array([2,3,4,5,6])


print(nn.mse(x,y))

df=pd.read_csv("winequality-red.csv")

# Split data into inputs and targets\
# inputs: 11
# targets: 1
targets=df['quality']
inputs=df.drop(columns='quality')

totalData=nn.formatData(df,.7,.29)
