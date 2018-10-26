import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

##################################################################################
################################ Classification example ##########################
################################## usinf Estimator API ###########################
##################################################################################

diabetes = pd.read_csv('pima-indians-diabetes.csv')
#print(diabetes.columns)

#################we have to normalize data################
#here label is class,
#remove group beacause it is String cannot normaize
#we convert age as categorical cols
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x-x.min()) / (x.max()-x.min() ))

##############3create continus feature cols###############
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

##########create non-continues features cols#############
# 2 ways
# 1) vocabulary list = for short # of category
# 2) hash bucket = for long # of category
assign_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A','B','C','D'])
#assign_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10) #maximum 10 but we good to to because it has only 4

#diabetes['Age'].hist(bins=20)
#plt.show()

# category the age
age_bucket = tf.feature_column.bucketized_column(age, boundaries = [20,30,40,50,60,70,80])

#put all together
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, assign_group, age_bucket]

################Train Test Split #########################
x_data = diabetes.drop('Class', axis = 1)
label = diabetes['Class']

x_train, x_eval, y_train, y_eval = train_test_split(x_data, label, test_size = 0.3,random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_eval,y=y_eval,batch_size=10,num_epochs=1,shuffle=False)

################### the model######################
#create model
#train model 
#evaluate model
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2) # 0 and 1 (2 classess)
model.train(input_fn=input_func, steps=1000) #train your model
result = model.evaluate(eval_input_func) # evalute your model
#print(result)

################### create predict function and predict ################
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=x_eval[:1], batch_size=10, num_epochs=1, shuffle=False)
predictions = model.predict(pred_input_func)
#print(list(predictions))
#print('\ndata\n')
#print(x_eval[:1])

######################### using DNN classifier model########################
embbedded_assing_group = tf.feature_column.embedding_column(categorical_column=assign_group,dimension=4) #in DNNClassifer we want to embbedded the
                                                                                                        #categorical cols
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, embbedded_assing_group, age_bucket]

x_train, x_eval, y_train, y_eval = train_test_split(x_data, label, test_size = 0.3,random_state=101)
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10], feature_columns=feat_cols,n_classes=2) # i want 3 layers 10 neurons each
dnn_model.train(input_func,steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_eval,y=y_eval,batch_size=10,num_epochs=1,shuffle=False)
result = dnn_model.evaluate(eval_input_func)

#print(result)





























