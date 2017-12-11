from __future__ import print_function

import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
c = tf.placeholder(tf.int16)
# Define some operations
#add = tf.add(a, b)
mul = tf.multiply(a, a)
add = tf.add(mul, b)
mult= tf.multiply(add, c)
# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(mult, feed_dict={a: 2, b: 3, c: 5}))
   # print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))
