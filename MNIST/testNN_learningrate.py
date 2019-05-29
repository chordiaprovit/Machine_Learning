import numpy as np
import collections
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time

data = pd.read_csv("/Users/provitchordia/Machine Learning/MNIST data/train.csv")

df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]

time_taken = []

train_set_score_sin = []
train_set_loss_sin = []
overall_accuracy_sin = []
learning_rate_sin =[]
test_set_score_sin = []

train_set_score = []
train_set_loss = []
overall_accuracy = []
learning_rate =[]
test_set_score = []

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.25)


def sinlayer(parm):
    nn = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(100),
                       random_state=1, learning_rate_init=parm)
    learning_rate_sin.append(parm)
    
    start_time = time.time()
    nn.fit(x_train, y_train)
    end_time = time.time()
    
    score = nn.score(x_train, y_train)
    loss = nn.loss_
    train_set_score_sin.append(score)
    train_set_loss_sin.append(loss)
    
    time_taken.append(end_time - start_time)
    
    pred = nn.predict(x_test)
    
    scoretest = nn.score(x_test, y_test)
    test_set_score_sin.append(scoretest)
    
    a = y_test.values
    
    # print collections.Counter(pred)
    
    correct_class = 0.0
    miss_class = 0.0
    
    for i in range(len(pred)):
        if pred[i] == a[i]:
            correct_class = correct_class + 1
    
    overall_accuracy_sin.append(correct_class / len(pred))
    print "--------"


sinlayer(0.1)
sinlayer(0.01)
sinlayer(0.001)
sinlayer(0.0001)

#print collections.Counter(df_y)
def twolayer(parm):
    nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(100,100),
                     random_state=1,learning_rate_init=parm)
    learning_rate.append(parm)
    
    start_time = time.time()
    nn.fit(x_train,y_train)
    end_time = time.time()
    
    score = nn.score(x_train, y_train)
    loss = nn.loss_
    train_set_score.append(score)
    train_set_loss.append(loss)
    
    
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

twolayer(0.1)
twolayer(0.01)
twolayer(0.001)
twolayer(0.0001)

print "train_set_score_sin", train_set_score_sin
print "test_set_score_sin", test_set_score_sin
print "train_set_loss_sin", train_set_loss_sin
print "overall_accuracy_sin", overall_accuracy_sin
print "learning_rate_sin", learning_rate_sin

print "train_set_score", train_set_score
print "test_set_score", test_set_score
print "train_set_loss", train_set_loss
print "overall_accuracy", overall_accuracy
print "time_taken", time_taken
print "learning_rate", learning_rate

plt.plot(learning_rate, train_set_score, label='training set score')
plt.plot(learning_rate, test_set_score, label='test set score')

plt.title('plot diffrent learning rate')
plt.xlabel("learning rate")
plt.ylabel("score")
plt.legend()

plt.show()



