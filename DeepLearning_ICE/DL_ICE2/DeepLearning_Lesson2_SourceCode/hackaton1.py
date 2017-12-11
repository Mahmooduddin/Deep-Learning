import sys, csv
import operator
import numpy as np
import tensorflow as tf


speaker = []
views = []
tags = []
ID = []



with open('TED_talk_small.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV)
    ID_sorted = sorted(readCSV, key=lambda x: int(x[0]))
    views_sorted = sorted(readCSV, key=lambda x: float(x[2]), reverse=True)
    speaker_sorted =

    for row in readCSV:
        ID1 = row[0]
        speaker1 = row[1]
        views1 = row[2]
        tags1 = row[3]
        ID.append(ID1)
        speaker.append(speaker1)
        tags.append(tags1)
        views.append(views1)

#tesnorflow graph
n_samples = readCSV.nrows - 1

X1 = tf.placeholder(tf.float32, name='X1')
X2 = tf.placeholder(tf.float32, name='X2')
X3 = tf.placeholder(tf.float32, name='X3')
Y = tf.placeholder(tf.float32, name='Y')

W1 = tf.Variable(0.0, name='w1')
W2 = tf.Variable(0.0, name='w2')
W3 = tf.Variable(0.0, name='w3')

Xmean = (X1 + X2 + X3) / 3

Y_predicted = (((X1 - Xmean) ** 2) * W1) + (((X2 - Xmean) ** 2) * W2) + (((X3 - Xmean) ** 2) * W3)

loss = tf.square(Y - Y_predicted, name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs/hackaton', sess.graph)

    for i in range(50):
        total_loss = 0
        for x1, x2, x3, y in readCSV:
            opt, l = sess.run([optimizer, loss], feed_dict={X1: x1, X2: x2, X3: x3, Y: y})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    writer.close()


def csv_find_ID(X):
    IDdex = ID.index(X)
    thespeaker = speaker[IDdex]
    theviews = views[IDdex]
    thetags = tags[IDdex]
    print("\n")
    print("For ID", X, "the speaker is", thespeaker, "and number of views is", theviews, "and tags are", thetags)
    print("\n")

def csv_find_speaker(X):
    speaker_list=[]

def csv_sorted_ID():
    for eachline in ID_sorted:
        print(eachline)


def csv_sorted_speaker():
    pass

def csv_sorted_views():
    print("a")
    print(views_sorted)
    for eachline in views_sorted:
        print(eachline)



def main():
    #print("1. Load the CSV file")
    print("\n")
    print("2. Find the properties with ID")
    print("3. Find the properties with speaker")
    print("4. Find the tags for video")
    print("5. Enter new video with speaker name, views and tags")
    print("6. Cluster the video related with tags")
    print("7. See the list sorted by ID")
    print("8. See the list sorted by Speaker")
    print("9. See the list sorted by number of views")
    print("1. Quit")
    print("\n")
    choice=input("Enter your selection (1-9) ?")

    if (choice == "2"):
        X = input(" Enter the ID number ? ")
        csv_find_ID(X)

    if (choice == "3"):
        X = input(" Enter the speaker name ? ")
        csv_find_speaker(X)


    if (choice == "4"):
        pass
    if (choice == "5"):
        pass
    if (choice == "6"):
        pass
    if (choice == "7"):
        csv_sorted_ID()

    if (choice == "8"):
        csv_sorted_speaker()

    if (choice == "9"):
        csv_sorted_views()

    if (choice=="1"):
        quit()


while True:
main()