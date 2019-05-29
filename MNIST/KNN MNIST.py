from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

mnist = fetch_mldata('MNIST original')

X, y = mnist.data, mnist.target


# take the MNIST data and construct the training and testing split, using 75% of the
# data for training and 25% for testing
(x_train, x_test, y_train, y_test) = train_test_split(np.array(mnist.data),
                                                                  mnist.target, test_size=0.25)


myList = list(range(1,30))
neighbors = filter(lambda x: x % 5 != 0, myList)
neigh = []


accuracies_def = []
accuracies_dist = []



# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 30, 5):
    # train the k-Nearest Neighbor classifier with the current value of `k`
    neigh.append(k)
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform', leaf_size=5)
    model.fit(x_train, y_train)
    
    # evaluate the model and update the accuracies list
    score = model.score(x_test, y_test)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies_def.append(score)
    print "score def", score
    
    model = KNeighborsClassifier(n_neighbors=k, weights='distance', leaf_size=5)
    model.fit(x_train, y_train)
    # evaluate the model and update the accuracies list
    score = model.score(x_test, y_test)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    
    accuracies_def.append(score)
    
    print "score dist", score
    print "-----"
    
# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies_def))
print("k=%d achieved highest accuracy with default distance of %.2f%% on validation data" % (myList[i],
                                                                       accuracies_def[i] * 100))

i = int(np.argmax(accuracies_dist))
print("k=%d achieved highest accuracy with eucledian distance of %.2f%% on validation data" % (myList[i],
                                                                       accuracies_dist[i] * 100))



# plot misclassification error vs k
plt.plot(neighbors, accuracies_def, label = "uniform dist")
plt.plot(neighbors, accuracies_dist, label = "euclid dist")
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy ')
plt.show()