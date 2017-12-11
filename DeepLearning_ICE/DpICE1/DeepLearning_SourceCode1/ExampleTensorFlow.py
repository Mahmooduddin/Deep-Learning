import tensorflow as tf

tf.InteractiveSession()
a=tf.zeros((2,2))
b=tf.ones((2,2))

print(tf.reduce_sum(b,reduction_indices=1).eval())

print(a.get_shape())

print(tf.reshape(a,(1,4)).eval())

print(tf.matmul(a,b).eval())