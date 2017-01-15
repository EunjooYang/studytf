import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1],(4,1))

print xy
print x_data
print y_data

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2,2],-1.0,1.0))
W2 = tf.Variable(tf.random_uniform([2,1],-1.0,1.0))

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Our hypothesis
L2 = tf.sigmoid(tf.matmul(X, W1)+b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2)+b2)

# cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

# Minimize
a = tf.Variable(0.01) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

'''
with tf.Session() as sess:
    sess.run(init)

    for step in xrange(100000):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 500 == 0:
            print (
                step,
                sess.run(cost, feed_dict={X:x_data, Y:y_data}),
                sess.run(W1),
                sess.run(W2)
            )

    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})
    print "Accuracy:", accuracy.eval({X:x_data, Y:y_data})
'''
