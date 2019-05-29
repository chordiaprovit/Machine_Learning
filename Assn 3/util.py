import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV


def getWineData():
    data = pd.read_csv("input/winequality-white.csv",sep=";")
    # data['quality'].replace(to_replace=[1, 2, 3, 4, 5], value=0, inplace=True)
    # data['quality'].replace(to_replace=[6, 7, 8, 9, 10], value=1, inplace=True)

    data['quality'].replace(to_replace=[3, 4], value=-1, inplace=True)
    data['quality'].replace(to_replace=[5, 6], value=0, inplace=True)
    data['quality'].replace(to_replace=[7, 8], value=1, inplace=True)
    print("Wine data :", data.shape)
    return data

def getAdultData():
    data = pd.read_csv("input/adultData.csv")

    print("data :", data.shape)
    data['occupation'] = data['occupation'].map({' ?': 0, ' Farming-fishing': 1, ' Tech-support': 2,
                                                 ' Adm-clerical': 3, ' Handlers-cleaners': 4, ' Prof-specialty': 5,
                                                 ' Machine-op-inspct': 6, ' Exec-managerial': 7,
                                                 ' Priv-house-serv': 8, ' Craft-repair': 9, ' Sales': 10,
                                                 ' Transport-moving': 11, ' Armed-Forces': 12, ' Other-service': 13,
                                                 ' Protective-serv': 14}).astype(int)
    data['income'] = data['income'].map({' <=50K\\': 0, ' >50K\\': 1}).astype(int)
    data['sex'] = data['sex'].map({' Male': 0, ' Female': 1}).astype(int)
    data['race'] = data['race'].map(
        {' Black': 0, ' Asian-Pac-Islander': 1, ' Other': 2, ' White': 3, ' Amer-Indian-Eskimo': 4}).astype(int)
    data['marital-status'] = data['marital-status'].map({' Married-spouse-absent': 0, ' Widowed': 1,
                                                         ' Married-civ-spouse': 2, ' Separated': 3, ' Divorced': 4,
                                                         ' Never-married': 5, ' Married-AF-spouse': 6}).astype(int)
    data['workclass'] = data['workclass'].map(
        {' Private': 1, ' Self-emp-inc': 2, ' State-gov': 3, ' Local-gov': 4, ' Without-pay': 5, ' Self-emp-not-inc': 6,
         ' Federal-gov': 7, ' Never-worked': 8, ' ?': 0}).astype(int)
    data['education'] = data['education'].map(
        {' 7th-8th': 0, ' Prof-school': 1, ' 1st-4th': 2, ' Assoc-voc': 3, ' Masters': 4, ' Assoc-acdm': 5,
         ' 9th': 6, ' Doctorate': 7, ' Bachelors': 8, ' 5th-6th': 9, ' Some-college': 10, ' 10th': 11,
         ' 11th': 12, ' HS-grad': 13, ' Preschool': 14, ' 12th': 15}).astype(int)
    data['relationship'] = data['relationship'].map(
        {' Wife': 0, ' Own-child': 1, ' Unmarried': 2, ' Husband': 3, ' Other-relative': 4,
         ' Not-in-family': 5}).astype(int)
    data.drop(['native-country'], axis=1, inplace=True)
    print("Adult data :", data.shape)

    return data


# def validationCurve(name,xLabel, est, X,Y, paramName, paramRange):
#     train_scores, test_scores = validation_curve(est, X, Y, param_name= paramName,param_range =paramRange,cv=5,scoring="accuracy",
#                                                  n_jobs=-1)
#
#     # Calculate mean and standard deviation for training set scores
#     train_mean = np.mean(train_scores, axis=1)
#     train_std = np.std(train_scores, axis=1)
#
#     # Calculate mean and standard deviation for test set scores
#     test_mean = np.mean(test_scores, axis=1)
#     test_std = np.std(test_scores, axis=1)
#
#     # Plot mean accuracy scores for training and test sets
#     plt.plot(paramRange, train_mean,'o-', label="Training score", color="black")
#     plt.plot(paramRange, test_mean,'o-', label="Cross-validation score", color="dimgrey")
#
#     # Plot accurancy bands for training and test sets
#     plt.fill_between(paramRange, train_mean - train_std, train_mean + train_std, color="gray")
#     plt.fill_between(paramRange, test_mean - test_std, test_mean + test_std, color="gainsboro")
#
#     # Create plot
#     plt.title(name)
#     plt.xlabel(xLabel)
#     plt.ylabel("Accuracy Score")
#     plt.tight_layout()
#     plt.legend(loc="best")
#     plt.grid()
#     plt.show()
#
#
# def gridSearch(est, paramGrid,X_train, y_train ):
#     clf_grid  = GridSearchCV(est, paramGrid,n_jobs=-1)
#     clf_grid.fit(X_train, y_train)
#
#     print("-----------------Original Features--------------------")
#     print("Best score: %0.4f" % clf_grid.best_score_)
#     print("Using the following parameters:")
#     print(clf_grid.best_params_)
#
#     return clf_grid.best_params_
#

# def learnigCurve(est,X_train,y_train,name):
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     # y_train = sc.transform(y_train)
#     est.fit(X_train, y_train)
#     train_sizes, train_scores, test_scores = learning_curve(est, X_train, y_train, cv=5,
#                                                             train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
#
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#
#     plt.figure()
#     plt.title(name)
#     plt.legend(loc="best")
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     plt.gca().invert_yaxis()
#
#     # box-like grid
#     plt.grid()
#
#     # plot the std deviation as a transparent range at each training set size
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
#                      color="g")
#
#     # plot the average training and test score lines at each training set size
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
#
#     # sizes the window for readability and displays the plot
#     # shows error from 0 to 1.1
#     plt.legend(loc=3)
#     plt.ylim(-.1, 1.1)
#     plt.show()
