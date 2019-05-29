import numpy as np
import collections
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import time

# data = pd.read_csv("/Users/provitchordia/Machine Learning/MNIST data/train.csv")
#
# df_x = data.iloc[:,1:]
# df_y = data.iloc[:,0]

time_taken = []
iters = []
train_set_score = []
train_set_loss = []
overall_accuracy = []
learning_rate =[]
activ_typ = []
test_set_score = []

mnist = fetch_mldata('MNIST original')
X, y = mnist.data, mnist.target
x_train, x_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.25)

#x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.25)

#print collections.Counter(df_y)

nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(100,100),
                 random_state=1)
activ_typ.append('logistic')

start_time = time.time()
nn.fit(x_train,y_train)
end_time = time.time()

score = nn.score(x_train, y_train)
loss = nn.loss_
train_set_score.append(score)
train_set_loss.append(loss)


time_taken.append(end_time - start_time)

pred=nn.predict(x_test)
scoretest = nn.score(x_test, y_test)
test_set_score.append(scoretest)

a=y_test.values


#print collections.Counter(pred)

correct_class=0.0
miss_class = 0.0

for i in range(len(pred)):
    if pred[i]==a[i]:
        correct_class = correct_class +1

overall_accuracy.append(correct_class/len(pred))
print "--------"


nn=MLPClassifier(activation='tanh',solver='sgd',hidden_layer_sizes=(100,100),
                 random_state=1)
activ_typ.append('tanh')

start_time = time.time()
nn.fit(x_train,y_train)
end_time = time.time()

score = nn.score(x_train, y_train)
loss = nn.loss_
train_set_score.append(score)
train_set_loss.append(loss)


time_taken.append(end_time - start_time)

pred=nn.predict(x_test)
scoretest = nn.score(x_test, y_test)
test_set_score.append(scoretest)

a=y_test.values


#print collections.Counter(pred)

correct_class=0.0
miss_class = 0.0

for i in range(len(pred)):
    if pred[i]==a[i]:
        correct_class = correct_class +1

overall_accuracy.append(correct_class/len(pred))
print "--------"


nn=MLPClassifier(activation='relu',solver='sgd',hidden_layer_sizes=(100,100),
                 random_state=1)
activ_typ.append('relu')

start_time = time.time()
nn.fit(x_train,y_train)
end_time = time.time()

score = nn.score(x_train, y_train)
loss = nn.loss_
train_set_score.append(score)
train_set_loss.append(loss)

time_taken.append(end_time - start_time)

pred=nn.predict(x_test)

scoretest = nn.score(x_test, y_test)
test_set_score.append(scoretest)
a=y_test.values


correct_class=0.0


for i in range(len(pred)):
    if pred[i]==a[i]:
        correct_class = correct_class +1

overall_accuracy.append(correct_class/len(pred))
print "--------"





print "activation_type", activ_typ
print "train_set_score", train_set_score
print "test_set_score", test_set_score
print "train_set_loss", train_set_loss
print "overall_accuracy", overall_accuracy
print "time_taken", time_taken

