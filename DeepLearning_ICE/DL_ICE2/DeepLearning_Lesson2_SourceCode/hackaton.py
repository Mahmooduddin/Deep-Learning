import sys, csv
import operator
from wsgiref.simple_server import make_server

speaker = []
views = []
tags = []
ID = []

with open('TED_talk_small.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV)
    for row in readCSV:
        ID1 = row[0]
        speaker1 = row[1]
        views1 = row[2]
        tags1 = row[3]
        ID.append(ID1)
        speaker.append(speaker1)
        tags.append(tags1)
        views.append(views1)

def csv_find_ID(X):
    IDdex = ID.index(X)
    thespeaker = speaker[IDdex]
    theviews = views[IDdex]
    thetags = tags[IDdex]
    print("\n")
    print("For ID", X, "the speaker is", thespeaker, "and number of views is", theviews, "and tags are", thetags)


def csv_find_speaker(X):
    speakerdex = speaker.index(X)
    theID = ID[speakerdex]
    theviews = views[speakerdex]
    thetags = tags[speakerdex]
    print("For speaker", X, "the ID is", theID, "and number of views is", theviews, "and tags are", thetags)
    print("\n")

def csv_new_record():

    speaker_name = input("\nEnter Speaker name ")
    views_name = input ("Enter number of views ")
    tags_name = input("Enter tags for video")
    ID_name = ID[-1]
    ID1 = int(ID_name) + 1
    ID.append(str(ID1))
    speaker.append(speaker_name)
    views.append(views_name)
    tags.append(tags_name)
    rows=zip(ID,speaker,views,tags)
    q=input("Do you want to save the file ? (y)")
    if q=="y":
        with open("TED_talk_small_new.csv", "w") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
    else:
        main()


def csv_sorted_ID():
    sample1 = open("TED_talk_small.csv", 'r')
    csv1 = csv.reader(sample1, delimiter=',')
    next(csv1)
    ID_sorted = sorted(csv1, key=lambda x: int(x[0]),reverse=False)
    for eachline in ID_sorted:
        print(eachline)

def csv_sorted_speaker():
    sample1 = open("TED_talk_small.csv", 'r')
    csv1 = csv.reader(sample1, delimiter=',')
    next(csv1)
    sortedlist = sorted(csv1, key=lambda row: row[1], reverse=False)
    for eachline in sortedlist:
        print(eachline)

def csv_sorted_views():
    sample1 = open("TED_talk_small.csv", 'r')
    csv1 = csv.reader(sample1, delimiter=',')
    next(csv1)
    ID_sorted = sorted(csv1, key=lambda x: int(x[2]), reverse=True)
    for eachline in ID_sorted:
        print(eachline)

def csv_sort_tag(X):
    for index,value in enumerate(tags):
        for j in value:
            if j==X:
                cul.append(index)
    print("Following Speakers and their Corresponding views are related to tag")
    for i in cul:
        print('Reviews: %r Speak Name %s '%(views(i),speaker(i)))
    #print(cul)


def main():
    print("1. Show the list")
    print("2. Find the video by ID")
    print("3. Find the video by speaker ")
    #print("4. Find the tags for video")
    print("5. Enter new video with speaker name, views and tags")
    print("6. Cluster the video related with tags")
    print("7. See the list sorted by ID")
    print("8. See the list sorted by Speaker")
    print("9. See the list sorted by number of views")
    print("11. Quit")
    print("\n")
    choice=input("Enter your selection (1-11) ?")

    if (choice == "1"):
        rows = zip(ID, speaker, views, tags)
        for r in rows:
            print(r)

    if (choice == "2"):
        X = input("Enter the ID number ? ")
        csv_find_ID(X)

    if (choice == "3"):
        X = input("Enter the speaker name ? ")
        csv_find_speaker(X)

    if (choice == "4"):
        pass

    if (choice == "5"):
        csv_new_record()

    if (choice == "6"):
        X = input(" Enter the the tag you want to search ? ")
        csv_sort_tag(X)

    if (choice == "7"):
        csv_sorted_ID()

    if (choice == "8"):
        csv_sorted_speaker()

    if (choice == "9"):
        csv_sorted_views()

    if (choice=="11"):
        quit()


while True:
main()

hackaton_TED()


def hackaton_TED(environ, start_response):
    status = '200 ok'
    headers = [('Content-type', 'text/plain')]
    start_response(status, headers)

    return

httpd = make_server('', 8005, hackaton_TED)

print ("Serving on porting 8005....")

httpd.serve_forever()

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
