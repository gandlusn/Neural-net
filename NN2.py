import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

x_data = np.linspace(0.0,10,1000000);
noise = np.random.randn(len(x_data));

y_true = (0.5*x_data) + 5 + 10*noise;

# first create the data frame for x and y
x_df = pd.DataFrame(data=x_data,columns=['X Data'])
y_df = pd.DataFrame(data=y_true,columns=['Y'])
# we use pandas to concat data
my_data = pd.concat([x_df,y_df],axis=1);

my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')


k = []
k

# we need to train batches of data

batch_size = 10

# first we are creating the variables for m and b with random values
m = tf.Variable(0.81);
b = tf.Variable(0.17);  

# now crearing a place holder all the batches of data points
xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size]) 

y_model = m*xph + b

error = tf.reduce_sum(tf.square(yph-y_model));

# now creating gradient descenrt with some learning rate to train models
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001);
train = optimizer.minimize(error);

# we have to intialize variables before using them
init = tf.global_variables_initializer();
fm =0;
fb =0;
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(init);
    
    epochs = 1000
    
    for i in range(epochs):
        # here we are goin to take batch sixze random indexes between 0 and len(x_data)
        rand_index = np.random.randint(len(x_data),size = batch_size)         
        feed = {xph:x_data[rand_index],yph: y_true[rand_index]}
        
        sess.run(train, feed_dict = feed);


    saver.save(sess,"new_dir/first_model.ckpt")

with tf.Session() as sess:
    saver.restore(sess,'new_dir/first_model.ckpt');
    fm, fb = sess.run([m,b])

print(fm,fb)

y_hat = x_data*fm+fb

my_data.sample(250).plot(kind='scatter',x='X Data',y = 'Y')

plt.plot(x_data,y_hat,'r')







