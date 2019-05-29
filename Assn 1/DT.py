from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
from numpy.random import RandomState
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import mean_squared_error

df = pd.read_csv("/Users/provitchordia/Machine Learning/Adult.csv")
print df.shape
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

# print df.CapitalGain.value_counts()
# print df.CapitalLoss.value_counts()

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

print df[:0]

#Since data is imbalanced, using stratify split to ensure data is splitted evenly in test and train sets
X_train, X_test, y_train, y_test = train_test_split(
    df, y_all, test_size=0.30, stratify=y_all, random_state=rs,)

#check to see of data is properly divided
#print pd.value_counts(pd.Series(y_train))
#print pd.value_counts(pd.Series(y_test))

depth = []
leaf = []
score_entropy = []
mse_entropy = []


max_depths = np.linspace(1, 50, 20, endpoint=True)


train_results = []
test_results = []

for max_depth in max_depths:
   depth.append(max_depth)
   dt = DecisionTreeClassifier(max_depth=max_depth)
   dt.fit(X_train, y_train)
   score = dt.score(X_train, y_train)
   train_results.append(score)
   
   y_pred = dt.predict(X_test)
   scoretest = dt.score(X_test, y_test)
   test_results.append(scoretest)
   
   mse = mean_squared_error(y_test, y_pred)
   mse_entropy.append(mse)
   
   print score
   print scoretest
   print mse
   print '----'
   

print "depth:", depth
print "train_results:", train_results
print "test_results", test_results
print "mse", mse_entropy

f1 = plt.figure(1)
plt.plot(max_depths, train_results, "b", label="Training set")
plt.plot(max_depths, test_results, "r", label="Test set ")
plt.plot(max_depths, mse_entropy, "g", label="MSE")
plt.ylabel("Accuracy score")
plt.xlabel("Tree depth")
plt.legend()
plt.title('Tree Depth to Accuracy')



depth = []
leaf = []
score_entropy1 = []
mse_entropy1 = []

min_samples_leafs = np.linspace(0.1, 0.5, 10, endpoint=True)
train_results1 = []
test_results1 = []

for min_samples_leaf in min_samples_leafs:
   dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
   leaf.append(min_samples_leaf)
   
   dt.fit(X_train, y_train)
   score = dt.score(X_train, y_train)
   train_results1.append(score)

   y_pred = dt.predict(X_test)
   scoretest = dt.score(X_test, y_test)
   test_results1.append(scoretest)

   mse = mean_squared_error(y_test, y_pred)
   mse_entropy1.append(mse)

   print score
   print scoretest
   print mse
   print '----'

print "leaf:", leaf
print "train_results:", train_results1
print "test_results", test_results1
print "mse", mse_entropy1


f2 = plt.figure(2)
plt.plot(leaf, train_results1, "b", label="Train Accuracy")
plt.plot(leaf, test_results1, "r", label="Test Accuracy")
plt.plot(leaf, mse_entropy1, "g", label="MSE")

plt.ylabel("Accuracy score")
plt.xlabel("min samples leaf")
plt.title('Leaf Size to Accuracy')
plt.legend()
plt.show()