from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

data = pd.read_csv("/Users/provitchordia/Machine Learning/MNIST data/train.csv")

df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]

(x_train, x_test, y_train, y_test) = train_test_split(df_x, df_y, test_size=0.30)

print len(x_train)
print len(x_test)

model = KNeighborsClassifier(n_jobs=2)

params = {'n_neighbors':[1,2,3,4,5],
          'leaf_size':[5,10,15],
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree','kd_tree'],
          'n_jobs':[2]}

#Making models with hyper parameters sets
model1 = GridSearchCV(model, param_grid=params, n_jobs=1)
#Learning
print "stating to fit"
model1.fit(x_train,y_train)
print "fit complete"
print ("score:", model1.score(x_test,y_test))
print model1.get_params(deep =True)
print "-----"
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction=model1.predict(x_test)

print("Accuracy:",metrics.accuracy_score(prediction,y_test))
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,y_test))