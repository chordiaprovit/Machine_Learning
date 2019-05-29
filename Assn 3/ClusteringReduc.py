from util import getWineData,getAdultData
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import  silhouette_score


def silhoutte(X,name):
    startTime = time.time()
    #print ('silhouette_avg startTime '), startTime

    #print ('data X'), X.shape

    silhouette_avg = []
    sse = []

    for i in range(2, 10):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=10, random_state=0)
        kmeans.fit(X)
        prediction = kmeans.fit_predict(X)
        silhouette_avg.append(silhouette_score(X, prediction))
        sse.append(kmeans.score(X))
        # hmg.append(metrics.homogeneity_score(Y, kmeans.labels_))

    y = [x / 1000000.0 for x in sse]
    SSE = (-pd.DataFrame(y))
    plt.figure()
    plt.plot(range(2, 10), SSE,'o-', range(2, 10), silhouette_avg,'o-')
    plt.grid()
    plt.xlabel("Number of K")
    plt.legend(['sse', 'silhouette_avg'])
    plt.title(name+":SSE and silhouette score")
    plt.show()



def distortion(X,name):
    startTime = time.time()
    #print ('distortion startTime '), startTime

    distortions = []

    for i in range(2, 10):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=10, random_state=0)
        kmeans.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        
    plot("Number of K","distortion score",name+':Elbow Method Analysis for Wine',distortions,range(2, 10))

    endTime = time.time()
    #print ('distortion endTime'), endTime


def plot(xLabel,ylabel,title,data,range):
    plt.plot(range, data, 'o-')
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def sse(X):
    startTime = time.time()
    #print ('SSE startTime '), startTime

    sse = []


    for i in range(2, 10):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=10, random_state=0)
        kmeans.fit(X)
        sse.append(kmeans.score(X))
    #print ('sse '), sse
    y = [x / 1000000.0 for x in sse]
    SSE = (-pd.DataFrame(y))
    # SSE.rename(columns=lambda x: x + ' SSE (left)', inplace=True)
    plot("Number of K", "SSE ", 'SSE for Wine', SSE, range(2, 10))

    endTime = time.time()
    #print ('sse endTime'), endTime


if __name__ == "__main__":
    data = getWineData()

    X = data.iloc[:, :-1]
    name = "Cluster"
    distortion(X,name)
    silhoutte(X,name)

    pcData = pd.read_csv("output file/wine_pca.csv",skiprows=1,usecols=range(1,8))
    name = "PCA"
    distortion(pcData,name)
    silhoutte(pcData,name)

    rfData = pd.read_csv("output file/wine_rf.csv",skiprows=1,usecols=range(1,4))
    X = rfData.iloc[:, :-1]
    name = "Random forest"
    distortion(X,name)
    silhoutte(X,name)

    rpData = pd.read_csv("output file/wine_rp.csv",skiprows=1,usecols=range(1,7))
    X = rpData.iloc[:, :-1]
    name = "Random Projection"
    distortion(X,name)
    silhoutte(X,name)

    icaData = pd.read_csv("output file/wine_ica.csv",skiprows=1,usecols=range(1,7))
    X = icaData.iloc[:, :-1]
    name = "ICA"
    distortion(X,name)
    silhoutte(X,name)