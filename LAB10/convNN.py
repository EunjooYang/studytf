import tensorflow as tf
import numpy as np
import input_data
import time

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w1, w2, w3, w4, w_out, p_keep_conv, p_keep_hidden):

    # l1ashape(?,28,28,32)
    # MNIST initial image size is 28*28, There are 32 filters
    # maxpooling & dropout process
    l1a = tf.nn.relu(tf.nn.conv2d(X,w1,strides=[1,1,1,1],padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    # l2a shape = (?, 14, 14, 64)
    # maxpooling & dropout process
    l2a = tf.nn.relu(tf.nn.conv2d(l1,w2,strides=[1,1,1,1],padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    # l3a shape=(?, 7, 7, 128)
    # maxpooling & dropout process
    # reshape to (?, 2048) & dropout process
    l3a = tf.nn.relu(tf.nn.conv2d(l2,w3,strides=[1,1,1,1],padding='SAME'))
    l3 = tf.nn.max_pool(l3a,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    l3 = tf.reshape(l3,[-1,w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3,w4))
    l4 = tf.nn.dropout(l4,p_keep_hidden)
    pyx = tf.matmul(l4 , w_out)
    return pyx

training_epoch = 15
display_step = 1
batch_size = 100

# MNIST VARIABLES INITIALIZE
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1,28,28,1) # 28x28x1 input image
teX = teX.reshape(-1,28,28,1) # 28x28x1 input image

# Input, Output
X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

# weights [row, column, depth, number of filter]
# output shoud be 10 classes (0~9)
w1 = init_weights([3,3,1,32])
w2 = init_weights([3,3,32,64])
w3 = init_weights([3,3,64,128])
w4 = init_weights([128*4*4,625])
w_out = init_weights([625,10])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w1, w2, w3, w4, w_out, p_keep_conv,p_keep_hidden)

# using Softmax & optimization process
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
init  = tf.initialize_all_variables()
predict_op = tf.argmax(py_x,1)

# Launch the graph in a session
start_time = time.time()
with tf.Session() as sess:

    tf.initialize_all_variables().run()

    for i in range(15):
        training_batch = zip(range(0, len(trX), batch_size),range(batch_size,len(trX),batch_size))
        for start, end in training_batch:

            sess.run(optimizer, feed_dict={X: trX[start:end], Y:trY[start:end],p_keep_conv:0.8, p_keep_hidden:0.5})
            #print 'cost:',
            #print sess.run(cost, feed_dict={X: trX[start:end], Y:trY[start:end],p_keep_conv:0.8, p_keep_hidden:0.5})
            sess.run(cost, feed_dict={X: trX[start:end], Y:trY[start:end],p_keep_conv:0.8, p_keep_hidden:0.5})

        test_indices = np.arange(len(teX))
        np.random.shuffle((test_indices))
        test_indices = test_indices[0:test_size]

        print 'error rate:',
        print(i, np.mean(np.argmax(teY[test_indices],axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         Y: teY[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))



end_time = time.time()
print 'elapsed time:', end_time-start_time
