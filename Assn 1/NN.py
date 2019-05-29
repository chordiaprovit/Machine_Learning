import numpy as np
import collections
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
import time
from sklearn.metrics import confusion_matrix

df = pd.read_csv("/Users/provitchordia/Machine Learning/Adult.csv")
#print df.shape
df.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]

#create a numpy.random.RandomState so that we can reproduce the same results each time
rs = RandomState(130917)

#convert the <=50Ks into 0 and the >50K into +1
df["Income"] = df["Income"].map({ " <=50K": 0, " >50K": 1 })

#extract the response variable into a numpy array and drop it from the dataframe:
y_all = df["Income"].values
df.drop("Income", axis=1, inplace=True,)

# #print df.CapitalGain.value_counts()
# #print df.CapitalLoss.value_counts()

# more than 95% of CapitalGain and CapitalLoss column consists of 0. So dropping the column
df.drop("CapitalGain", axis=1, inplace=True,)
df.drop("CapitalLoss", axis=1, inplace=True,)

#Education and EducationNum represents the same thing, its just the numeric representation
#so dropping Education as EducationNum is numeric
df.drop("Education", axis=1, inplace=True,)

#Convert the Age, fnlwgt, EducationNum and HoursPerWeek to floating point
df.Age = df.Age.astype(float)
df.fnlwgt = df.fnlwgt.astype(float)
df.EducationNum = df.EducationNum.astype(float)
df.HoursPerWeek = df.HoursPerWeek.astype(float)

#one-hot encoding to transform them into numerical features
df = pd.get_dummies(df, columns=[
    "WorkClass", "MaritalStatus", "occupation", "Relationship",
    "Race", "Gender", "NativeCountry",
])

#check if data is imbalanced
#print pd.value_counts(pd.Series(y_all))

#only 24% of data is in single class, so using stratify

#print df[:0]

#Since data is imbalanced, using stratify split to ensure data is splitted evenly in test and train sets
X_train, X_test, y_train, y_test = train_test_split(
    df, y_all, test_size=0.30, stratify=y_all, random_state=rs,)

#check to see of data is properly divided
#print pd.value_counts(pd.Series(y_train))
#print pd.value_counts(pd.Series(y_test))

time_taken = []
iters = []
learing_rate = []
train_set_score = []
test_set_score = []
train_set_loss = []
overall_accuracy = []



max_iters = np.arange(10, 200, 10)
#print max_iters
lrs = np.arange(0.1, 1, 0.1 )
#print lrs


# for lr in lrs:
#     nn = MLPClassifier(activation='relu', solver='sgd', hidden_layer_sizes=(10),
#                        random_state=1, learning_rate_init=lr,max_iter = 10)
#     learing_rate.append(lr)
#
#     start_time = time.time()
#     nn.fit(X_train, y_train)
#     end_time = time.time()
#
#     pred = nn.predict(X_test)
#     score = nn.score(X_train, y_train)
#     loss = nn.loss_
#
#     scoretest = nn.score(X_test, y_test)
#
#     test_set_score.append(scoretest)
#     train_set_score.append(score)
#     train_set_loss.append(loss)
#
#     time_taken.append(end_time - start_time)
#     #print score
#     #print scoretest
#     #print "--------"
#
# #print "learning rate", learing_rate
# #print "train_set_score", train_set_score
# #print "test_set_score", test_set_score
# #print "train_set_loss", train_set_loss
# #print "time_taken", time_taken
# # #print "--------"





time_taken = []
iters = []

train_set_score1 = []
test_set_score1 = []
train_set_loss1 = []
train_set_score = []
test_set_score = []
train_set_loss = []
overall_accuracy = []

for max_iter in max_iters:
    nn = MLPClassifier(activation='relu', solver='sgd', hidden_layer_sizes=(12),
                       random_state=1, max_iter = max_iter)
    iters.append(max_iter)
    nn.fit(X_train, y_train)

    pred = nn.predict(X_test)
    score = nn.score(X_train, y_train)
    loss = nn.loss_

    scoretest = nn.score(X_test, y_test)

    test_set_score1.append(scoretest)
    train_set_score1.append(score)
    train_set_loss1.append(loss)
    #print score
    #print scoretest
    #print "--------"

#print "iters", iters
#print "train_set_score1", train_set_score1
#print "test_set_score1", test_set_score1
#print "train_set_loss1", train_set_loss1

iters = []


for max_iter in max_iters:
    nn = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(12,2),
                       random_state=1, max_iter = max_iter)
    iters.append(max_iter)

    start_time = time.time()
    nn.fit(X_train, y_train)
    end_time = time.time()

    pred = nn.predict(X_test)
    score = nn.score(X_train, y_train)
    loss = nn.loss_

    scoretest = nn.score(X_test, y_test)

    test_set_score.append(scoretest)
    train_set_score.append(score)
    train_set_loss.append(loss)

    time_taken.append(end_time - start_time)
    #print score
    #print scoretest
    #print "--------"

#print "iters", iters
#print "train_set_score", train_set_score
#print "test_set_score", test_set_score
#print "train_set_loss", train_set_loss
#print "time_taken", time_taken


plt.plot(iters, train_set_score1, label='train score 1 layer')
plt.plot(iters, test_set_score1, label='test score 1 layer')
plt.plot(iters, train_set_loss1, label = "loss 1 layer")
plt.plot(iters, train_set_score, label='train score 2 layer')
plt.plot(iters, test_set_score, label='test score 2 layer')
plt.plot(iters, train_set_loss, label = "loss 2 layer")


plt.title('NN varying iters for diffrent layers (128 nuerons)')
plt.xlabel("iters")
plt.ylabel("score/loss")
plt.legend()
plt.show()



