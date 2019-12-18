from __future__ import print_function
import tensorflow as tf
import numpy as np
from numpy import *
import os
import matplotlib.pyplot as plt
import time

learning_rate = 0.05
training_epochs = 10000
display_step = 50
layer_1_nodes = 30
layer_2_nodes = 30
layer_3_nodes = 30
output_nodes  = 1

# tf Graph Input

# tf Graph Input
X =  tf.placeholder(tf.float32,[None,2])
Y =  tf.placeholder(tf.float32,[None,1])



layer_1 = tf.Variable(tf.random_normal([2,layer_1_nodes]))
layer_1_bias = tf.Variable(tf.random_normal([layer_1_nodes]))
layer_2 = tf.Variable(tf.random_normal([layer_1_nodes,layer_2_nodes]))
layer_2_bias = tf.Variable(tf.random_normal([layer_2_nodes]))
layer_3 = tf.Variable(tf.random_normal([layer_2_nodes,layer_3_nodes]))
layer_3_bias = tf.Variable(tf.random_normal([layer_3_nodes]))

output_layer = tf.Variable(tf.random_normal([layer_3_nodes,output_nodes]))
output_layer_bias = tf.Variable(tf.random_normal([output_nodes]))

layer_1_output = tf.nn.sigmoid(tf.matmul(X,layer_1)+layer_1_bias)
layer_2_output = tf.nn.sigmoid(tf.matmul(layer_1_output,layer_2)+layer_2_bias)
layer_3_output = tf.nn.sigmoid(tf.matmul(layer_2_output,layer_3)+layer_3_bias)
pred  = tf.matmul(layer_3_output,output_layer)+output_layer_bias

# Train Data
#train_X = np.array([[1.0,2.0],[3.0,4.0],[5.0,6.0]])

train_X = np.genfromtxt('Input.tex', delimiter=',')

n_samples = train_X.shape[0]


train_Y = reshape(np.genfromtxt('Target.tex', delimiter=','),[n_samples,1])


cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            x = reshape(x,[1,x.shape[0]])
            y = reshape(y,[1,y.shape[0]])
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost,'\n')

    layer_1_vals           = sess.run(layer_1)
    layer_1_bias_vals      = sess.run(layer_1_bias)
    
    layer_2_vals           = sess.run(layer_2)
    layer_2_bias_vals      = sess.run(layer_2_bias)
    
    layer_3_vals           = sess.run(layer_3)
    layer_3_bias_vals      = sess.run(layer_3_bias)
        
    output_layer_vals      = sess.run(output_layer)
    output_layer_bias_vals = sess.run(output_layer_bias)

# Writing the optimal parameters on files.
    
np.savetxt('layer_1_vals.txt',layer_1_vals)
np.savetxt('layer_2_vals.txt',layer_2_vals)
np.savetxt('layer_3_vals.txt',layer_3_vals)
np.savetxt('output_layer_vals.txt',output_layer_vals)
np.savetxt('layer_1_bias_vals.txt',layer_1_bias_vals)
np.savetxt('layer_2_bias_vals.txt',layer_2_bias_vals)
np.savetxt('layer_3_bias_vals.txt',layer_3_bias_vals)
np.savetxt('output_layer_bias_vals.txt',output_layer_bias_vals)
