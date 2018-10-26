import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, classification_report


# class that able to create a data and generate batches to send it back

class TimeSeriesData () :
    def __init__(self, num_points, xmin, xmax) :
        self.xmin = xmin
        self.xmax  = xmax
        self.num_points = num_points
        self.resolution = (xmax-xmin) / num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    #  comparing data that we don't have
    def ret_true(self, x_series) :
        return(np.sin(x_series))

    # genarate batches of data
    def next_batch(self, batch_size, steps, return_batch_ts = False):
        # grab a random starting point for each batch
        rand_start = np.random.rand(batch_size,1)

        # convert to be on time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps*self.resolution))
        
        # create batch time series on the x axis
        batch_ts = ts_start + np.arange(0.0, steps+1) * self.resolution
        
        # create y data for the time series x axis from previous steps
        y_batch = np.sin(batch_ts)

        # formatting for RNN
        if return_batch_ts :
            return y_batch[:,:-1].reshape(-1,steps,1) , y_batch[:,1:].reshape(-1,steps,1), batch_ts
        else :
            return y_batch[:,:-1].reshape(-1,steps,1) , y_batch[:,1:].reshape(-1,steps,1)
        

ts_data = TimeSeriesData(250,0,10) # 250 points between 0 and 10

num_time_steps = 30

y1, y2, ts = ts_data.next_batch(1,num_time_steps, True)

# plt.plot(ts.flatten()[1:], y2.flatten(), '*', label='single traing instance') #flatten means take everythinf in one list
# plt.plot(ts_data.x_data, ts_data.y_true, label='sin(t)')
# plt.legend()
# plt.tight_layout() 

# plt.show()

# trainin Data
train_inst = np.linspace(5,5+ts_data.resolution*(num_time_steps+1), num_time_steps+1)

#plt.title('A Training Instance')
#plt.plot(train_inst[:-1],ts_data.ret_true(train_inst[:-1]), 'bo', markersize=15, alpha=0.5,label='INSTANCE')
# predict one time step
#plt.plot(train_inst[1:],ts_data.ret_true(train_inst[1:]), 'ko', markersize=5, alpha=0.5,label='Target')
#plt.show()

################ create the model ####################

tf.reset_default_graph()
num_input = 1 # basically one feature in ts
num_neurons = 100 # 100 neurons in one layer
num_outputs = 1
learning_rate = 0.01
num_train_iterations = 2000
batch_size = 1

# palceholders
x = tf.placeholder(tf.float32,[None, num_time_steps,num_input])
y = tf.placeholder(tf.float32,[None, num_time_steps,num_outputs])

# RNN cell layer
#BasicLSTMCell, GRUCell, MultiRNNCell
cell = tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size= num_outputs)

# get output and states of these basic RNN cells
outputs, states = tf.nn.dynamic_rnn(cell=cell,inputs= x, dtype=tf.float32)

#lost function and optimizer
# MSE as loss function
loss = tf.reduce_mean(tf.square(outputs-y))

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

#Session
saver = tf.train.Saver() # save the model

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for iteration in range(num_train_iterations) :
        x_batch, y_batch = ts_data.next_batch(batch_size,num_time_steps)
        sess.run(train, feed_dict={x: x_batch, y:y_batch})

        if iteration % 100 == 0 :
            mse = loss.eval(feed_dict={x: x_batch, y:y_batch})
            print(iteration, "\tMSE", mse)
        
    saver.save(sess,"./rnn_time_series_model_codealong")


with tf.Session() as sess :
    saver.restore(sess, "./rnn_time_series_model_codealong")

    x_new = np.sin(np.array(train_inst[:-1].reshape(-1,num_time_steps,num_input)))
    y_pred = sess.run(outputs, feed_dict={x:x_new})

#graph
#traing instance
plt.plot(train_inst[:-1],np.sin(train_inst[:-1]), "bo", markersize=15,alpha=0.5,label='trainig_instance')

#target to predect (correct test valus np.sin(train))
plt.plot(train_inst[1:], np.sin(train_inst[1:]),"ko", markersize=10,label="target")

#model predection
plt.plot(train_inst[1:], y_pred[0,:,0],'r.', markersize=10, label='predection')
plt.xlabel('TIME')
plt.legend()
plt.tight_layout()

plt.show()



