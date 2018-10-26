import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, classification_report

# read the monthly-milk-production.csv file and index_col is month
milk = pd.read_csv('monthly-milk-production.csv', index_col='Month')

#make the index a time series 
milk.index = pd.to_datetime(milk.index)

#milk.plot()
#plt.show()

################### Train test split ########################
# Let's attempt to predect a year's worth of data (12 month or 12 steps in future)
# create a test train split using indexing

train_set = milk.head(156)
test_set = milk.tail(12)


##################### Scale the data #########################
# use sklearn.preprocessing to scale the data using MinMaxScaler
# Remember to only fit_transform on the training data then transform the test data
# you shouldn't fit on the test data as well
# otherwise you are assuming you know the future behavior

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)


################### Batch function #############################
# we will need a function that can feed batches of the training data

def next_batch(training_data, batch_size, steps) :
    


























































