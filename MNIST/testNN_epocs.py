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
iters = []
train_set_score = []
test_set_score = []
train_set_loss = []
overall_accuracy = []



x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.30)

#print collections.Counter(df_y)

def single_layer(parm):
    nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(100),
                     random_state=1,max_iter=parm)
    iters.append(parm)
    
    start_time = time.time()
    nn.fit(x_train,y_train)
    end_time = time.time()
    
    score = nn.score(x_train, y_train)
    loss = nn.loss_
    scoretest = nn.score(x_test, y_test)
    test_set_score.append(scoretest)
    train_set_score.append(score)
    train_set_loss.append(loss)
    
    
    time_taken.append(end_time - start_time)
    
    pred=nn.predict(x_test)
    
    a=y_test.values
    
    
    #print collections.Counter(pred)
    
    correct_class=0.0
    miss_class = 0.0
    
    for i in range(len(pred)):
        if pred[i]==a[i]:
            correct_class = correct_class +1
    
    overall_accuracy.append(correct_class/len(pred))
    print "--------"

single_layer(20)
single_layer(50)
single_layer(100)
single_layer(150)
single_layer(200)


#==================

iters_multi = []
train_set_score_multi = []
test_set_score_multi = []
train_set_loss_multi = []
overall_accuracy_multi = []

def twolayer(parm):
    nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(100,100),
                     random_state=1,max_iter=20)
    iters_multi.append(parm)
    
    start_time = time.time()
    nn.fit(x_train,y_train)
    end_time = time.time()
    
    score = nn.score(x_train, y_train)
    loss = nn.loss_
    scoretest = nn.score(x_test, y_test)
    test_set_score_multi.append(scoretest)
    train_set_score_multi.append(score)
    train_set_loss_multi.append(loss)
    
    
    time_taken.append(end_time - start_time)
    
    pred=nn.predict(x_test)
    
    a=y_test.values
    
    
    #print collections.Counter(pred)
    
    correct_class=0.0
    miss_class = 0.0
    
    for i in range(len(pred)):
        if pred[i]==a[i]:
            correct_class = correct_class +1
    
    overall_accuracy_multi.append(correct_class/len(pred))
    print "--------"
    
    
    nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(100,100),
                     random_state=1,max_iter=50)
    iters_multi.append(50)
    
    start_time = time.time()
    nn.fit(x_train,y_train)
    end_time = time.time()
    
    score = nn.score(x_train, y_train)
    loss = nn.loss_
    scoretest = nn.score(x_test, y_test)
    test_set_score_multi.append(scoretest)
    train_set_score_multi.append(score)
    train_set_loss_multi.append(loss)
    
    time_taken.append(end_time - start_time)
    
    pred=nn.predict(x_test)
    
    a=y_test.values
    
    correct_class=0.0
    
    for i in range(len(pred)):
        if pred[i]==a[i]:
            correct_class = correct_class +1
    
    overall_accuracy_multi.append(correct_class/len(pred))
    print "--------"

twolayer(20)
twolayer(50)
twolayer(100)
twolayer(150)
twolayer(200)



plt.plot(iters, train_set_score, label='training set score')
plt.plot(iters, test_set_score, label='test set score')
plt.plot(iters, train_set_score_multi, label='training set score(multi layer)')
plt.plot(iters, test_set_score_multi, label='test set score(multi layer)')
plt.title('Accuracy at diffrent epocs and # of layers')
plt.xlabel("# of epocs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.plot(iters, train_set_loss, label='train_set_loss')
plt.plot(iters, train_set_loss_multi, label='train_set_loss(multi layer)')
plt.title('Loss at diffrent epocs and # of layers')
plt.xlabel("# of epocs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print "iters",iters
print "train_set_score", train_set_score
print "test_set_score", test_set_score
print "train_set_loss", train_set_loss
print "overall_accuracy", overall_accuracy
print "train_set_score_multi", train_set_score_multi
print "test_set_score_multi", test_set_score_multi
print "train_set_loss_multi", train_set_loss_multi
print "overall_accuracy_multi", overall_accuracy_multi

print "time_taken", time_taken
