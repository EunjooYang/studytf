import tensorflow as tf

# Simple hello world using Tensorflow
# The op is added as a node to the defuault graph
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')

print hello

# Start tf session
sess = tf.Session()
print sess.run(hello)
