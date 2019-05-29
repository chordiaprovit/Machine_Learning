from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score
import pandas as pd
from numpy.random import RandomState

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
##print pd.value_counts(pd.Series(y_all))

#only 24% of data is in single class, so using stratify

#print df[:0]

#Since data is imbalanced, using stratify split to ensure data is splitted evenly in test and train sets
X_train, X_test, y_train, y_test = train_test_split(
    df, y_all, test_size=0.30, stratify=y_all, random_state=rs,)

#check to see of data is properly divided
##print pd.value_counts(pd.Series(y_train))
##print pd.value_counts(pd.Series(y_test))

parm1= []
parm2 = []
rec = []
f1s = []
prec = []
acc = []


estimators = [10, 50, 100, 200, 300, 400, 500 ]
samples = [5, 10, 15, 20]

for estimator in estimators:
    parm1 = []
    parm2 = []
    rec = []
    f1s = []
    prec = []
    acc = []
    train_score = []
    for sample in samples:
        clf = AdaBoostClassifier(n_estimators=estimator,
                                 algorithm='SAMME.R',
                                 base_estimator=DecisionTreeClassifier(criterion='entropy',
                                                                       min_samples_split=sample))
        clf.fit(X_train, y_train)
        label_predict = clf.predict(X_test)
        
        precision = precision_score(y_test, label_predict, average='weighted')
        
        from sklearn.metrics import recall_score
        recall = recall_score(y_test, label_predict, average='weighted')
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        
        accuracy = accuracy_score(y_test, label_predict)
        # #print('estimator size:', estimator_size)
        
        parm1.append(estimator)
        parm2.append(sample)
        prec.append(str(precision))
        rec.append(str(recall))
        f1s.append(str(f1))
        acc.append(str(accuracy))
    #print "estimator:", parm1
    #print "sample:", parm2
    #print "precision:", prec
    #print "recall:", rec
    #print "f1 score:", f1s
    #print "accuracy:", acc
    #print "----"

