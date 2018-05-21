import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#HRLPER

# INIT Weightm
def init_weights(shape):
    inti_random_dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inti_random_dist);
#iNIT Bias 
def init_bias(shape):
    init_bias = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias)
# Con2D
def conv2D(x,W):
    #x input tensor --> [batch, H, W, channels]
    #W Kernel-->[filter Height,filter width, CHannels IN, Channels Out]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#Pooling 2*2
def max_pool_2by2(x):
    ##x input tensor --> [batch, H, W, channels]
    #ksize is dimension of widow for each value in pooling result
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#Pooling 2*2
def max_pool_4by4(x):
    ##x input tensor --> [batch, H, W, channels]
    #ksize is dimension of widow for each value in pooling result
    return tf.nn.max_pool(x,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")

#Pooling 2*2
def max_pool_2by2_stride(x):
    ##x input tensor --> [batch, H, W, channels]
    #ksize is dimension of widow for each value in pooling result
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,3,3,1],padding="SAME")

#Convolutional layer
def convolutional_layer(input_x,shape):
    W = init_weights(shape)
    bias = init_bias([shape[3]])
    return tf.nn.relu(conv2D(input_x,W)+bias)

#normal fully connected
def normal_full(input_layer,size):
    input_size=int(input_layer.get_shape()[1])
    W = init_weights([input_size,size])
    b = init_bias([size])
    return tf.matmul(input_layer,W)+b

def Batch_normal_full(input_layer,phase,scope):
    with tf.variable_scope(scope):
            h1 = tf.contrib.layers.fully_connected(x, 100, 
                                                   activation_fn=None,
                                                   scope='dense')
            h2 = tf.contrib.layers.batch_norm(h1, 
                                              center=True, scale=True, 
                                              is_training=phase,
                                              scope='bn')
            return tf.nn.relu(h2, 'relu')


#Placeholders
x = tf.placeholder(tf.float32,shape=[220,220])
y_true = tf.placeholder(tf.float32,shape=[None,1])

#Layer1 {CONVOLUTION}
x_image = tf.reshape(x,[-1,220,220,1])
convo_1 = convolutional_layer(x_image,shape=[7,7,1,32])
# here 5,5 is the patch size  is the no of input channels so here 1 channel because of grey scale image
#32 is the no of features we are computing, so it is the no of output channels
convo_1_pooling = max_pool_4by4(convo_1)


#layer2 {CONVOLUTION}
convo_2 = convolutional_layer(convo_1_pooling,shape=[7,7,32,64])
convo_2_pooling = max_pool_2by2_stride(convo_2)

#Layer3 {CONVOLUTION}
convo_3 = convolutional_layer(convo_2_pooling,shape=[5,5,64,70])
convo_3_pooling = max_pool_2by2(convo_3)

#Layer4 {CONVOLUTION}
convo_4 = convolutional_layer(convo_3_pooling,shape=[5,5,70,70])
convo_4_pooling = max_pool_2by2(convo_4)

#Layer5 {FlATTENING}
# now we will flatten out the whole out put and present it ot th.e fukly connected layer
convo_4_flat = tf.reshape(convo_4_pooling,[-1,7*7*70])
# here 7*7 because the of the 2*2 pooling we applied 2 times so each dimension heoght and widht of the channels will shrink by 4
#28*28 becomes 7*7 and 64 is th no of output channels in last convolutional layer

#Layer6 {Fully Connecting}
full_layer_1 = tf.nn.relu(normal_full(convo_4_flat,2000))

#Layer6 {dropout} 
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_1,keep_prob=hold_prob)

#Layer7 {Fully Connected}
y_pred = tf.nn.relu(normal_full(full_one_dropout,1))


#LOSS function
cross_entropy =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

steps = 1000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(steps):
        
        batch_x, batch_y = mnist.train.next_batch(40)
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        
        if i%100 ==0:
            print("ON STEP : {}",format(i))
            print("Accuracy : ")
            correct_prediction = tf.equal(tf.arg_max(y_pred,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0}))
            print("\n")







