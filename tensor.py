import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

hello = tf.constant("hello ")
world  = tf.constant("world")

with tf.Session() as sess :
    result = sess.run(hello+world)

a = tf.constant(10)
b = tf.constant(20)

with tf.Session() as sess:
    result = sess.run(a+b) 

#numpy ways in tensorflow
constant = tf.constant(10)
fill_mat = tf.fill((4,4),10)
myzeros = tf.zeros((4,4))
myones = tf.ones((4,4))

#random numbers normal distribution
myrandn = tf.random_normal((4,4),mean=0.0,stddev=1.0)

#random numbers uniform distribution
myrandu = tf.random_uniform((4,4),minval=0, maxval=1)

op = [constant,fill_mat,myzeros,myones,myrandn,myrandu]
#interactive session
sess = tf.InteractiveSession()

# for i in op:
#     print(sess.run(i))
#     print('\n')

#matrix mul
a = np.arange(1,5).reshape((2,2))
b = np.array([10,100]).reshape((2,1))
matA = tf.constant(a)
matB = tf.constant(b)
#print(sess.run(tf.matmul(matA,matB)))
#print(tf.matmul(matA,matB).eval())

#graph
n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1+n2

#print(tf.get_default_graph())
g = tf.Graph()
graph_one = tf.get_default_graph()
graph_two = tf.Graph()

#set graph_two as default graph
with graph_two.as_default() :
    #print(graph_two is tf.get_default_graph()) #print true
    pass


#variables and placeholder
#2 main types of tensor object in graph (variable and placeholder)
#variables can hold the values of weight and biases throughout the session (should be initilized)
#placehoder used to feed tha actual traning (initial empty)

my_tensor = tf.random_uniform((4,4),0,1)
my_var = tf.Variable(initial_value=my_tensor)
#we need to initilized variable before run it
sess.run(tf.global_variables_initializer())
#print(sess.run(my_var))

#placeholder
ph = tf.placeholder(tf.float32,shape=(None,5))

############    building models  ###############
np.random.seed(101) #np rand seed "same result for everyone"
tf.set_random_seed(101) #tf rand seed

rand_a = np.random.uniform(0,100,(5,5))
rand_b = np.random.uniform(0,100,(5,1))

#create placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

#create oparation
add_op = a+b
mul_op = a*b

#create session => they can use graph with feed dictionary
with tf.Session() as sess :
    add_result = sess.run(add_op, feed_dict={a:rand_a,b:rand_b})
    #print(add_result)
    mult_result = sess.run(mul_op,feed_dict={a:rand_a,b:rand_b})
    #print(mult_result)

#create neurons nw
n_features = 10
n_dens_neurons = 3 #assumtion : 1 layer and 3 dense neurons

x = tf.placeholder(tf.float32,shape=(None,n_features))  #rows are #of sample
                                                        # cols are # of features

w = tf.Variable(tf.random_normal([n_features,n_dens_neurons])) #weight
b = tf.Variable(tf.ones([n_dens_neurons]))

#build the perception model
wx = tf.matmul(x,w)
z = tf.add(wx,b)

#acivated fn
a = tf.sigmoid(z)

#init tha Variable
init = tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(init)
    layer_out = sess.run(a,feed_dict={x:np.random.random([1,n_features])})
    #print(layer_out)

#############    simple Regreession ex    ###########
x_data = np.linspace(0,10,10)+np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10)+np.random.uniform(-1.5,1.5,10)

#plt.plot(x_data,y_label, '*')
#plt.show()

# y = mx+b
m = tf.Variable(0.44) #initial random values
b = tf.Variable(0.87)

#cost fn
err = 0
for x,y in zip(x_data, y_label) :
    y_hat = m*x+b #here represent the predected value
    err += (y-y_hat)**2 #this is what we want to minize

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) #To minize the err
train = optimizer.minimize(err)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    trainig_steps = 1 #ajust it 
    for i in range(trainig_steps) :
        sess.run(train)
        final_slop, final_intercept = sess.run([m,b])

x_test = np.linspace(-1,11,10)
#y = mx+b
y_pred_plot = final_slop*x_test + final_intercept #normalized data

#        plot
#plt.plot(x_data,y_label, '*')
#plt.plot(x_test,y_pred_plot, '*') 
#plt.show()

###########################################################################
############################ Regrssion analysis ###########################
###########################################################################
x_data = np.linspace(0.0,10, 1000000)
noise = np.random.randn(len(x_data))

# create model
#y = mx + b ; m=.5 b = 5
y_true = (0.5*x_data) + 5 + noise   #adding noice we can have complex patten
                                    # we do not use it when we have real data set to feed

#create data frame
x_df = pd.DataFrame(data=x_data, columns=['X data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])
#print(x_df.head())

#concatenate 2 data frams
my_data = pd.concat([x_df,y_df],axis=1)
#print(my_data.head())

# note = 
# if we plot all data it cause to crash the kernal beacuse there are hugh data set
# it is good practise to plot sample from it 

#my_data.sample(n=250).plot(kind='scatter', x='X data', y='Y')
#plt.show()

# need to train data in rensorflow model batch by batch
batch_size = 8
m = tf.Variable(0.81)
b = tf.Variable(0.17)

xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

y_model = m*xph + b

error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    batches = 1000

    for i in range(batches) :
        rand_index = np.random.randint(len(x_data), size=batch_size,)
        feed = {xph:x_data[rand_index], yph:y_true[rand_index]}
        sess.run(train, feed_dict=feed)
    
    model_m, model_b = sess.run([m,b])

y_hat = model_b*x_data + model_b # estimate y_hat

#print('b = ',model_b)
#print('m = ',model_m)

######################################################################
################ Regression Analysis using ##############################
############## Estimator API ############################################
#########################################################################

# steps :
# define the list of feature cols
# define the Estimator model
# create data input function
# call train evaluate model and predect method on the estimate object

feat_cols = [tf.feature_column.numeric_column(key='X',shape=[1])] #all the feature in list
Estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols) #use estimator as linear regression

x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size = 0.3,random_state=101) #use x_train 70% and x_eval 30% of data

input_func = tf.estimator.inputs.numpy_input_fn({'X':x_train},y_train, batch_size=8, num_epochs=None, shuffle=True)
input_input_func = tf.estimator.inputs.numpy_input_fn({'X':x_train},y_train, batch_size=8, num_epochs=1000, shuffle=False)
eval_input_func = tf.estimator.inputs.numpy_input_fn({'X':x_eval},y_eval, batch_size=8, num_epochs=1000, shuffle=False)

#now trian the estimator(using  matrix)
Estimator.train(input_fn=input_func, steps=1000) #step just like num_epochs
trian_matrix =Estimator.evaluate(input_fn=input_input_func, steps=1000)#get matrix on the training data
eval_matrix = Estimator.evaluate(input_fn=eval_input_func, steps=1000) #basically give lost fn for me

print('My Training data matrix')
print(trian_matrix)

print('My Test data matrix')
print(eval_matrix)
#note
#your training data matrix is much better that evaluation matrix data
#then you must be overfitting your train data
# Overfitting refers to a model that models the training data too well.
# Underfitting refers to a model that can neither model the training data nor generalize to new data.

#predict data here
brand_new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'X':brand_new_data}, shuffle=False)
list(Estimator.predict(input_fn_predict))
predictions = []
for pred in Estimator.predict(input_fn_predict):
    predictions.append(pred['predictions'])

print(predictions)

my_data.sample(n=250).plot(kind='scatter', x='X data', y='Y')
plt.plot(brand_new_data,predictions,'r*')
plt.show()