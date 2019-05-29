from sklearn.metrics import silhouette_samples, silhouette_score
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from util import getAdultData, getWineData
from sklearn import metrics
from scipy.spatial.distance import cdist

data = getWineData()

dataX = data.iloc[:,:-1]
dataY = data.iloc[:,-1]

clusters =  [2,3,4,5,6,7,8,9,10]

distortions = []
# time1 = []
# ARSk = []
# AMISk = []
# HSk = []
# CSk = []
# VMSk = []
Sil = []
SSE = []
for k in clusters:
    start_time = time.time()

    kmeanModel = KMeans(n_clusters=k, max_iter = 500, random_state=0,init="k-means++").fit(dataX)
    kmeanModel.fit(dataX)
    prediction = kmeanModel.fit_predict(dataX)
    end_time = time.time()

    sil_avg = silhouette_score(dataX, prediction)
    # ARS = metrics.adjusted_rand_score(dataY, prediction)
    # AMIS = metrics.adjusted_mutual_info_score(dataY, prediction)
    # HS = metrics.homogeneity_score(dataY, prediction)
    # CS = metrics.completeness_score(dataY, prediction)
    # VMS = metrics.v_measure_score(dataY, prediction)
    sse = (kmeanModel.score(dataX) * -1) / 100000000000000.0


    # ARSk.append(ARS)
    # AMISk.append(AMIS)
    # HSk.append(HS)
    # CSk.append(CS)
    # VMSk.append(VMS)
    Sil.append(sil_avg)
    SSE.append(sse)

    cls_time = end_time - start_time
    # time1.append(cls_time)
    distortions.append(sum(np.min(cdist(dataX, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / dataX.shape[0])

# print("time:", time1)
# print("Adjusted Random Score:", ARSk)
# print("AMIS:", AMISk)
# print("HS:", HSk)
# print("CS:", CSk)
# print("VMS:", VMSk)
print("Sil:", Sil)
print("SSE:", SSE)
# Plot the elbow
# plt.plot(clusters, AMISk, label = "Mutual info")
# plt.plot(clusters, ARSk, label = "Adjusted Random")
# plt.plot(clusters, HSk, label = "homogeneity")
# plt.plot(clusters, CSk, label = "completeness")
# plt.plot(clusters, VMSk, label = "v_measure")
# plt.plot(clusters, distortions, "bx-")

plt.plot(clusters,SSE, label = "Sum of squared error")
plt.plot(clusters,Sil, label = "Silhouette Average")

plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
plt.ylabel('Score')
plt.title('Different score showing the optimal k')
plt.legend()
plt.grid()
plt.show()

plt.plot(clusters,distortions, label = "distortions Average")
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.legend()
plt.grid()
plt.show()

