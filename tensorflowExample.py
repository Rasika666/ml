import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, classification_report

##############################################################################
########################### Regression Exercise ##############################
##############################################################################

#import the cal_housing_clean.csv file with pandas.
#separate into training(70%) and testing(30%)
housing = pd.read_csv('cal_housing_clean.csv')
y_val = housing['medianHouseValue']
x_data = housing.drop('medianHouseValue', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_val, test_size = 0.3,random_state=101)

############### scale the feature data ################
# use sklearn preprocessing to create a MinMaxScaler for feature data
# Fit this scaler only to the train data.
#Then use it to transform X_Test and X_Train
#Then use the scaled x_test and X_train along with pd.
#Dataframe to re-create two dataframes of scaled data

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = pd.DataFrame(data=scaler.transform(x_train),columns=x_train.columns,index=x_train.index) #reset x_train scale version of data
x_test = pd.DataFrame(data=scaler.transform(x_test),columns=x_test.columns,index=x_test.index)

###################### create feature columns #####################
# create the necessary tf.feature_column for estimator
# they should all be trated as continuous numeric_columns
#print(housing.columns)
age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
householders = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')
feat_cols = [age, rooms, bedrooms, pop, householders, income]

#############################input function #####################
# create the input function for the estimator object
# play ariund with the batch-size and num-epochs
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train,batch_size=10, num_epochs=1000, shuffle=True)

######################### model ###################################
#create the Estimater model use a DNNRegrssor.
#play around with the hidden units
model = tf.estimator.DNNRegressor(hidden_units=[6,6,6], feature_columns=feat_cols)

########################### Train model ###############################
# Train your model for ~1000 steps
model.train(input_fn=input_func, steps=10)

############################## predict input function #################
# create a predict input  function and use the predict method off your estimator model
# to create a list or predictions on your test data
predict_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=10, num_epochs=1, shuffle=False)
pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)

################################# RMSE ############################
# you should be able to get round 100,000 RMSE
#calculate using sklearn

final_pred = []
for pred in predictions :
    final_pred.append(pred['predictions'])

#print(mean_squared_error(y_test,final_pred)**0.5)



##############################################################################
########################### classification Exercise ##############################
##############################################################################
 # test to predict what class income they belong in 

#################### The Data ##################################
# Read the census_data.csv data with pandas
census = pd.read_csv('census_data.csv')

# Tensorflow won't be able to understand String as labels
# you need to use pandas .apply() to mapply a custom function that converts 1s and 0s
#print(census['income_bracket'].unique())

def label_fix(label) :
    if label == ' <=50K' :
        return 0
    else :
        return 1

census['income_bracket'] = census['income_bracket'].apply(label_fix)

################### perform Train Test split on Data #####################
x_data = census.drop('income_bracket', axis=1)
y_label = census['income_bracket']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_label, test_size = 0.3,random_state=101)
  
################## create feature columns for tf.estimator #################
#print(census.columns)

#categorical cols
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ['Female', 'Male'])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_state = tf.feature_column.categorical_column_with_hash_bucket("marital_status",hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass" ,hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country",hash_bucket_size=1000)

#continuous feature cols
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

feat_cols = [gender, occupation, marital_state, education, relationship, workclass, native_country, age, education_num, capital_gain,capital_loss, hours_per_week]


######################## create input Function ##############################
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train, batch_size=100, num_epochs=None, shuffle=True)

####################### create a model ##############################
model = tf.estimator.LinearClassifier(feature_columns=feat_cols)

################# Train you model at least 5000 steps #######################
model.train(input_fn=input_func, steps=100)

################### Evaluate the model #############################
# create prediction input function only suport for x data
predict_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=len(x_test), shuffle=False)

#use model.predict() and pass in your input fn
# This will produse a genarator of predictions
#which you can then transform into a list
pred_gen = model.predict(input_fn=predict_input_func)
predictions = list(pred_gen)

final_pred = [pred['class_ids'][0] for pred in predictions]
#print(final_pred[:5])

##################### classification Report ########################
# you can figure out how to use it to get easily full report of your model's performance on test data

#print(classification_report(y_test,final_pred))




















