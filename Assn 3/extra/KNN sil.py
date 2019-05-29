from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from util import getAdultData

data = getAdultData()

wineX = data.iloc[:,:-1]
wineY = data.iloc[:,-1]

#wineX = StandardScaler().fit_transform(X)

range_n_clusters = [2,3,4,5,6,7,8]

figures = []
avg = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    prediction = clusterer.fit_predict(wineX)

    # Create a subplot with 1 row and 2 columns
    sil_avg = silhouette_score(wineX, prediction)
    print ("For number of clusters: " + str(n_clusters) + "; average sil score:" + str(sil_avg))
    avg.append(sil_avg)

plt.plot(range_n_clusters, avg, "bx-")

plt.xlabel('k')
plt.ylabel('silhouette score')
plt.title('silhouette score for different K')
plt.legend()
plt.grid()
plt.show()