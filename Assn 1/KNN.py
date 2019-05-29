from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from numpy.random import RandomState
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_mldata
import time

df = pd.read_csv("/Users/provitchordia/Machine Learning/Adult.csv")
print df.shape
df.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]

# create a numpy.random.RandomState so that we can reproduce the same results each time
rs = RandomState(130917)

# convert the <=50Ks into -1 and the >50K into +1
df["Income"] = df["Income"].map({" <=50K": -1, " >50K": 1})

# extract the response variable into a numpy array and drop it from the dataframe:
y_all = df["Income"].values
df.drop("Income", axis=1, inplace=True, )

# print df.CapitalGain.value_counts()
# print df.CapitalLoss.value_counts()

# more than 95% of CapitalGain and CapitalLoss column consists of 0. So dropping the column
df.drop("CapitalGain", axis=1, inplace=True, )
df.drop("CapitalLoss", axis=1, inplace=True, )

# Convert the Age, fnlwgt, EducationNum and HoursPerWeek to floating point
df.Age = df.Age.astype(float)
df.fnlwgt = df.fnlwgt.astype(float)
df.EducationNum = df.EducationNum.astype(float)
df.HoursPerWeek = df.HoursPerWeek.astype(float)

# one-hot encoding to transform them into numerical features
df = pd.get_dummies(df, columns=[
    "WorkClass", "Education", "MaritalStatus", "occupation", "Relationship",
    "Race", "Gender", "NativeCountry",
])

# check if data is imbalanced
pd.value_counts(pd.Series(y_all))

print df[:0]

# Since data is imbalanced, using stratify split to ensure data is splitted evenly in test and train sets
X_train, X_test, y_train, y_test = train_test_split(
    df, y_all, test_size=0.30, stratify=y_all, random_state=rs, )
print X_train.shape
print X_test.shape

myList = list(range(1, 30))
neighbors = filter(lambda x: x % 5 != 0, myList)
neigh = []

accuracies_def = []
accuracies_dist = []
roc_auc1 = []
roc_auc2 = []
# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 50, 4):
    # train the k-Nearest Neighbor classifier with the current value of `k`
    neigh.append(k)
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform', leaf_size=5)
    model.fit(X_train, y_train)
 
    # evaluate the model and update the accuracies list
    score = model.score(X_test, y_test)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies_def.append(score)
    print "score def", score
    y_pred = model.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_curve(y_test, y_pred)
    roc_auc_uni = auc(false_positive_rate, true_positive_rate)
    roc_auc1.append(roc_auc_uni)
    
    model = KNeighborsClassifier(n_neighbors=k, weights='distance', leaf_size=5)
    model.fit(X_train, y_train)
    # evaluate the model and update the accuracies list
    score = model.score(X_test, y_test)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    
    accuracies_dist.append(score)
    
    print "score dist", score
    
    y_pred = model.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_curve(y_test, y_pred)
    roc_auc_dis = auc(false_positive_rate, true_positive_rate)
    roc_auc2.append(roc_auc_dis)
    print "-----"

# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies_def))
print("k=%d achieved highest accuracy with default distance of %.2f%% on validation data" % (myList[i],
                                                                                             accuracies_def[i] * 100))

i = int(np.argmax(accuracies_dist))
print("k=%d achieved highest accuracy with eucledian distance of %.2f%% on validation data" % (myList[i],
                                                                                               accuracies_dist[
                                                                                                   i] * 100))

print roc_auc1
print roc_auc2

#plot misclassification error vs k
f1 = plt.figure(1)
plt.plot(neigh, accuracies_def, label="uniform dist")
plt.plot(neigh, accuracies_dist, label="euclid dist")
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy ')
plt.title("Num of neighbours to test accuracy")
plt.legend()

f2 = plt.figure(2)
plt.plot(neigh, roc_auc1, label="AUC uniform dist")
plt.plot(neigh, roc_auc2, label="AUC euclid dist")
plt.xlabel('Number of Neighbors K')
plt.ylabel('AUC score ')
plt.title("Num of neighbours to AUC score")
plt.legend()

plt.show()