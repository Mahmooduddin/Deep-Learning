""" Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from
the number of fire in the city of Chicago
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/Iris.xls'

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1
print(n_samples)

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X1 = tf.placeholder(tf.float32, name='X1')
X2 = tf.placeholder(tf.float32, name='X2')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
W1 = tf.Variable(0.0, name='w1')
W2 = tf.Variable(0.0, name='w2')

# Step 4: build model to predict Y
Y_predicted = X1 * W1 + X2 * W2

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000001).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    # Step 8: train the model
    for i in range(50):  # train the model 50 epochs
        total_loss = 0
        for x1, x2, y in data:
            # Session runs train_op and fetch values of loss
            opt, l = sess.run([optimizer, loss], feed_dict={X1: x1, X2 : x2, Y: y})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # close the writer when you're done using it
    writer.close()

    # Step 9: output the values of w and b
    W1, W2 = sess.run([W1, W2])
print(W1,W2)

# plot the results
X1, X2, Y = data.T[0], data.T[1], data.T[2]
Y_predic=X1 * W1 + X2 * W2
X1 = X1/X1.max(axis=0)
X2 = X2/X2.max(axis=0)
Y = Y/Y.max(axis=0)

Y_predic= Y_predic/Y_predic.max(axis=0)

plt.plot(X1, X2, Y, 'bo', label='Real data')
plt.plot(X1, X2, Y_predic, 'r', label='Predicted data')
#plt.plot(X2, Y, 'bo', label='Real data')
#plt.plot(X2, Y_predic, 'r', label='Predicted data')

plt.legend()
plt.show()