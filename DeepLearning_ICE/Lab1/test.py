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
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')
#Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
W = tf.Variable(0.0, name='W')
b = tf.Variable(0.0, name='b')

# Step 4: build model to predict Y
pred = tf.nn.softmax(tf.matmul(x,W)+b)

# Step 5: use the square error as the loss function
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

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
    W, b = sess.run([W, b])
#print(W1,W2)

x, y = data.T[0], data.T[1]
plt.plot(x, y, 'bo', label='Real data')
plt.plot(x, pred, 'r', label='Predicted data')
plt.legend()
plt.show()