import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.random_projection import SparseRandomProjection
from util import getAdultData,getWineData
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import time
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse as sps
from scipy.linalg import pinv



def rp(x):
    if x == 'wine':
        data = getWineData()

    if x == 'adult':
        data = getAdultData()

    dataX = data.iloc[:,:-1]

    wineX = StandardScaler().fit_transform(dataX)

    print('wineX ',wineX.shape)
    tmp = []
    tmp1 =[]

    for comp in range(1,12):
        rp = SparseRandomProjection(n_components = comp)
        tmp.append(pairwiseDistCorr(rp.fit_transform(wineX),wineX))
        rp.fit(wineX)
        # tmp1.append(reconstructionError(rp, wineX))

    print('tmp ',tmp)
    title = x+' : Randomised Projection'
    plot('# component','DistCorrelation',title,tmp,range(1,12))
    # plot('# component', 'ReconstructionError', title, tmp1, range(1, 12))


def NN(x,comp):
    if x == 'wine':
        data = getWineData()

    if x == 'adult':
        data = getAdultData()

    dataX = data.iloc[:, :-1]
    dataY = data.iloc[:, -1]

    wineX = StandardScaler().fit_transform(dataX)

    print('wineX ', wineX.shape)
    tmp = []

    rp = SparseRandomProjection(n_components=comp)
    X_rp = pd.DataFrame(rp.fit_transform(wineX))
    X_rp.to_csv("output file/"+ x+'_rp.csv')

    X_train, X_test, y_train, y_test = train_test_split(X_rp, dataY, test_size=0.30, stratify=dataY)
    print(X_train.shape)
    print(X_test.shape)

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
    plt.title(x+": RP Learning Curve Nueral Network Classifier")
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

def plot(xLabel,ylabel,title,data,range):
    plt.plot(range, data, 'o-', color="grey")
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def pairwiseDistCorr(x1,x2):
    d1 = pairwise_distances(x1)
    d2 = pairwise_distances(x2)
    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]

def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)

    reconstructed = ((p*W)*(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)

if __name__ == "__main__":
    rp('wine')
    rp('adult')
    NN('wine',6)
    NN('adult', 9)