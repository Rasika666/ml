import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, classification_report

#import data files
saleData = pd.read_csv('File 01 - Sales Data.csv')
#print(saleData.head)
print(saleData.columns)

#normalizing data
cols_to_norm = []

#set x_data and columns
y_data = saleData['Retailer Code']
x_data = saleData.drop('')