import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from util import getAdultData,getWineData
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import time



def pcaWine():

    data = getWineData()
    dataX = data.iloc[:,:-1]
    dataY = data.iloc[:,-1]

    wineX = StandardScaler().fit_transform(dataX)

    pca = PCA(random_state=0)
    pca.fit(wineX)

    X_pca = pca.transform(wineX)
    print("original shape:   ", wineX.shape)
    print("transformed shape:", X_pca.shape)

    tmp1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    print(tmp1)

    plotVariancePCA(tmp1,'Number of Components','Cumulative Explained Variance','Wine: Explained Variance vs Number of Components')

    plotEigenValuePCA(pca.explained_variance_, 'Principal Components', 'Eigenvalues', 'Wine: Eigenvalues vs Principal Components(Scree)')
    pca = PCA(n_components=7)
    pca.fit(wineX)
    X_pca = pd.DataFrame(pca.transform(wineX))
    print (type(X_pca))
    X_pca.to_csv('output file/wine_pca.csv')

    nueralNets(X_pca, dataY, 'Wine')


def pcaAdult():
    data = getAdultData()
    dataX = data.iloc[:, :-1]
    dataY = data.iloc[:, -1]

    adultX = StandardScaler().fit_transform(dataX)

    pca = PCA(random_state=0)
    pca.fit(adultX)

    X_pca = pca.transform(adultX)
    print("original shape:   ", adultX.shape)
    print("transformed shape:", X_pca.shape)

    tmp1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
    print(tmp1)

    plotVariancePCA(tmp1, 'Number of Components', 'Cumulative Explained Variance',
                    'Adult: Explained Variance vs Number of Components')

    plotEigenValuePCA(pca.explained_variance_, 'Principal Components', 'Eigenvalues',
                      'Adult: Eigenvalues vs Principal Components(Scree)')
    pca = PCA(n_components=10)
    pca.fit(adultX)
    X_pca = pd.DataFrame(pca.transform(adultX))
    print (type(X_pca))
    X_pca.to_csv('output file/adult_pca.csv')

    nueralNets(X_pca, dataY, 'Adult')

def plotVariancePCA(data,xlabel,ylabel,title):

     plt.plot(data)
     plt.xlabel(xlabel)
     plt.ylabel(ylabel)
     plt.title(title)
     plt.grid()
     plt.show()

def plotEigenValuePCA(data,xlabel,ylabel,title):
     plt.plot(data)
     plt.xlabel(xlabel)
     plt.ylabel(ylabel)
     plt.title(title)
     plt.grid()
     plt.show()


def nueralNets(X_pca,dataY,label):
    X_train, X_test, y_train, y_test = train_test_split(X_pca, dataY, test_size=0.30, stratify=dataY)
    print(X_train.shape)
    print(X_test.shape)

    start_time = time.time()
    nueralNets = MLPClassifier(activation='tanh', solver='sgd', hidden_layer_sizes=(10),
                       random_state=5, learning_rate_init=0.1)
    nueralNets.fit(X_train, y_train)

    train_sizes, train_scores, test_scores = learning_curve(nueralNets, X_train, y_train, cv=10,
                                                            train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(label+" :Learning Curve Nueral Network Classifier")
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
    plt.title(label+" :Validation Curve With Nueral network")
    plt.xlabel("Max Iters")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    end_time = time.time()

    print("VC", end_time - start_time)


if __name__ == "__main__":
    pcaWine()
    pcaAdult()