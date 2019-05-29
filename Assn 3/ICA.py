from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from util import getWineData
from util import getAdultData
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import time


def adult():
    data = getAdultData()
    dataX = data.iloc[:, :-1]
    dataY = data.iloc[:, -1]
    adultX = StandardScaler().fit_transform(dataX)

    loss = []
    # Adult
    dims = range(1, 12)
    kurt = {}
    for dim in dims:
        ica = FastICA(n_components=dim)
        tmp = ica.fit_transform(adultX)

        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt[dim] = tmp.abs().mean()

        wine_ica = ica.fit_transform(adultX)
        wine_proj = ica.inverse_transform(wine_ica)

        loss.append(((adultX - wine_proj) ** 2).mean())

    kurt = pd.Series(kurt)
    # print(kurt.shape)
    # kurt.to_csv('wine_kurt.csv')
    #


    plt.plot(kurt)
    plt.xlabel('Independent Components')
    plt.ylabel('Kurtosis')
    plt.title('Adult: Kurtosis ICA')
    plt.grid()
    plt.show()

    plt.plot(loss)
    plt.xlabel('Independent Components')
    plt.ylabel('loss')
    plt.title('Adult: negentropy ICA')
    plt.grid()
    plt.show()
    #
    ica = FastICA(n_components=7, random_state=5)
    X1 = ica.fit_transform(adultX)
    X1 /= X1.std(axis=0)

    print("original shape:   ", adultX.shape)
    print("transformed shape:", X1.shape)
    X_ica = pd.DataFrame(ica.transform(adultX))
    X_ica.to_csv('output file/adult_ica.csv')
    name = "adult"
    nueralNets(X_ica, dataY,name)

def whiteWine():
    data = getWineData()
    #data = getAdultData()
    dataX = data.iloc[:,:-1]
    dataY = data.iloc[:,-1]
    wineX = StandardScaler().fit_transform(dataX)

    loss = []
    # Wine
    dims = range(1,12)
    kurt = {}
    for dim in dims:
        ica = FastICA(n_components=dim)
        tmp = ica.fit_transform(wineX)

        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt[dim] = tmp.abs().mean()


        wine_ica = ica.fit_transform(wineX)
        wine_proj = ica.inverse_transform(wine_ica)

        loss.append(((wineX - wine_proj) ** 2).mean())

    kurt = pd.Series(kurt)
    # print(kurt.shape)
    # kurt.to_csv('wine_kurt.csv')
    #

    plt.plot(kurt)
    plt.xlabel('Independent Components')
    plt.ylabel('Kurtosis')
    plt.title('Wine: Kurtosis ICA')
    plt.grid()
    plt.show()
    #
    plt.plot(loss)
    plt.xlabel('Independent Components')
    plt.ylabel('loss')
    plt.title('Wine: negentropy ICA')
    plt.grid()
    plt.show()
    #
    ica = FastICA(n_components=6, random_state=5)
    X1 = ica.fit_transform(wineX)
    X1 /= X1.std(axis=0)

    print("original shape:   ", wineX.shape)
    print("transformed shape:", X1.shape)
    X_ica = pd.DataFrame(ica.transform(wineX))
    X_ica.to_csv('output file/wine_ica.csv')
    name ="wine"
    nueralNets(X_ica,dataY,name)

def nueralNets(dataX,dataY,name):
    X_train, X_test, y_train, y_test = train_test_split(dataX , dataY, test_size=0.30, stratify=dataY)
    print(X_train.shape)
    print(X_test.shape)

    start_time = time.time()
    NN = MLPClassifier(activation='tanh', solver='sgd', hidden_layer_sizes=(10),
                           random_state=5,  learning_rate_init = 0.1)
    NN.fit(X_train, y_train)

    train_sizes, train_scores, test_scores = learning_curve(NN, X_train, y_train, cv=10,
                                                            train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("ICA Learning Curve for " +name + ":Nueral Network Classifier")
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
    train_scores, test_scores = validation_curve(clf, X_train, y_train, cv=10,param_name="max_iter",
                                                     param_range=param_range,scoring="accuracy",n_jobs=1)

    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, "o-", label="Training score",  color="blue")
    plt.plot(param_range, test_mean, "o-", label="Cross-validation score", color="green")

    # Create plot
    plt.title("Validation Curve With Nueral network")
    plt.xlabel("Max Iters")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    end_time = time.time()

if __name__ == "__main__":
    whiteWine()
    adult()