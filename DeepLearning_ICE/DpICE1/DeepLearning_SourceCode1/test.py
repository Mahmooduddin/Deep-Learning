from __future__ import print_function

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(5)
d = ((a ** 2) + b) * c

sess = tf.Session()

print(sess.run(d))