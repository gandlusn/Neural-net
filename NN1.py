import tensorflow as tf
import numpy as np
n_features = 3;
n_outputs = 3;

x = tf.placeholder(tf.float32,(None,n_features))

w = tf.Variable(tf.random_normal([n_features,n_outputs]))#these are the weights between input layer and output

b = tf.Variable(tf.ones([n_outputs]))# these are the biases for the output layer

xw = tf.matmul(x,w);#multiplying weights    

z = tf.add(xw,b) # adding biases in the end

a = tf.sigmoid(z); # applying sigmoid function on it

init = tf.global_variables_initializer(); # we intailze all variables before we run

with tf.Session() as sess:
    sess.run(init);
    layer_out= sess.run(a,feed_dict={x:np.random.random([1,n_features])})
    print(layer_out);

#Simple Regression Example

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10) # randomly choosing 10 numbers between 0 and 10 and adding noise to them by adding and substracting 1.5,-1.5

y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

import matplotlib.pyplot as plt

plt.plot(x_data,y_label,'*');

m= tf.Variable(0.44)#assign some random values 
b = tf.Variable(0.87)
error =0;
for x,y in zip(x_data,y_label):
    y_hat = m*x+b
    error += (y-y_hat)**2 

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.001);
train = optimizer.minimize(error);# using to gradient discent to minimize the error between x and y


init = tf.global_variables_initializer(); # we intailze all variables before we run

with tf.Session() as sess:
    sess.run(init);
    training_steps= 1;
    for i in range(training_steps):# here training steps are epochs
        sess.run(train);# every time  it will run through all the data points
    slope, intercept = sess.run([m,b])


x_test = np.linspace(-1,11,10) # these are the x points between 1 and 11
y = slope*x_test + intercept;# here we are using y = m*x + c equation to get the correct line
plt.plot(x_test,y);
plt.plot(x_data,y_label,'*');
plt.show()