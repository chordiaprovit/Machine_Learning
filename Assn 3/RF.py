import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from util import getWineData,getAdultData
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import pandas as pd



def rfWine():

    wine = getWineData()
    wX = wine.iloc[:,:-1]
    wineY = wine.iloc[:,-1]
    wineX = StandardScaler().fit_transform(wX)

    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
    fs_wine = rfc.fit(wineX,wineY).feature_importances_

    indices = np.argsort(fs_wine)[::-1]
    print(indices)

    plot('Feature importances of Wine',range(wineX.shape[1]),fs_wine[indices],indices,'Feature number for Wine dataset','Scores')

    important_indices = [10,7,4]
    print('important_indices ',important_indices)

    pd.DataFrame(wineX[:, important_indices]).to_csv('output file/wine_rf.csv')

    X_train, X_test, y_train, y_test = train_test_split(wineX, wineY, test_size=0.30, stratify=wineY)

    train_important = X_train[:, important_indices]
    test_important = X_test[:, important_indices]

    # print('train_important ',train_important)
    # print('test_important ', test_important)
    print(train_important.shape)
    print(test_important.shape)

    NN(train_important,test_important,y_train,y_test,'wine')


def NN(X_train,X_test,y_train,y_test,x):

    start_time = time.time()
    NN = MLPClassifier(activation='tanh', solver='sgd', hidden_layer_sizes=(10),
                       random_state=5, learning_rate_init=0.1)
    NN.fit(X_train, y_train)

    train_sizes, train_scores, test_scores = learning_curve(NN, X_train, y_train, cv=10,
                                                            train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(x+": Learning Curve Nueral Network Classifier")
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()

    # box-like grid
    plt.grid()

    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")

    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    # sizes the window for readability and displays the plot
    # shows error from 0 to 1.1
    plt.legend()
    plt.ylim(-.1, 1.1)
    plt.show()
    end_time = time.time()

    print("LC", end_time - start_time)

    start_time = time.time()
    clf = MLPClassifier(activation='tanh', solver='sgd')
    param_range = np.arange(10, 500, 30)
    train_scores, test_scores = validation_curve(clf, X_train, y_train, cv=10, param_name="max_iter",
                                                 param_range=param_range, scoring="accuracy", n_jobs=1)

    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, "o-", label="Training score", color="blue")
    plt.plot(param_range, test_mean, "o-", label="Cross-validation score", color="green")

    # Create plot
    plt.title(x+" :Validation Curve With Nueral network")
    plt.xlabel("Max Iters")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    end_time = time.time()

    print('end_time ',end_time)

def plot(title, dataX, dataY,limit,xLabel,yLabel):

    # Plot the feature importances of the forest
    plt.figure()
    plt.title(title)
    plt.bar(dataX, dataY,color="b",  align="center")
    plt.xticks(dataX, limit)
    # plt.xlim([-1, dataX])
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()


def rfAdult():
    adult = getAdultData()
    aX = adult.iloc[:, :-1]
    adultY = adult.iloc[:, -1]
    adultX = StandardScaler().fit_transform(aX)

    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
    fs_adult = rfc.fit(adultX,adultY).feature_importances_

    indices = np.argsort(fs_adult)[::-1]
    print(indices)
    plot('Feature importances of Adult', range(adultX.shape[1]), fs_adult[indices], indices,
         'Feature number for Adult dataset', 'Scores')

    important_indices = [0, 2, 5]
    print('important_indices ', important_indices)

    pd.DataFrame(adultX[:, important_indices]).to_csv('output file/adult_rf.csv')

    X_train, X_test, y_train, y_test = train_test_split(adultX, adultY, test_size=0.30, stratify=adultY)

    train_important = X_train[:, important_indices]
    test_important = X_test[:, important_indices]

    # print('train_important ',train_important)
    # print('test_important ', test_important)
    print(train_important.shape)
    print(test_important.shape)

    NN(train_important, test_important, y_train, y_test, 'adult')


if __name__ == "__main__":
    rfWine()
    rfAdult()