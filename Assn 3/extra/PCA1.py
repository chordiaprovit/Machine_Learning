import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from util import getWineData
from util import getAdultData
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import time


df = getWineData()
#data = getAdultData()

dataX = df.iloc[:,:-1]
dataY = df.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.3, random_state=0)

# sc = StandardScaler()
# X_train_std = sc.fit_transform(X_train)
# X_test_std = sc.fit_transform(X_test)
#
# covariant_matrix = np.cov(X_train_std.T)
# eigen_values, eigen_vectors = np.linalg.eig(covariant_matrix)
#
# tot = sum(eigen_values)
# var_exp = [(i / tot) for i in sorted(eigen_values, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)
#
# plt.bar(range(1,14), var_exp, alpha=0.5, align='center',
#                   label='individual explained variance')
# plt.step(range(1,14), cum_var_exp, where='mid',
#                   label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc='best')
# plt.show()
dfx = pd.DataFrame(data=dataX)
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
dfx_pca = pca.fit(dfx)
dfx_trans = pca.transform(dfx)
print(dfx_trans)

plt.figure(figsize=(10,6))
plt.scatter(dfx_trans[0],dfx_trans[1],c=df['quality'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Class separation using first two principal components\n",fontsize=20)
plt.xlabel("Principal component-1",fontsize=15)
plt.ylabel("Principal component-2",fontsize=15)
plt.show()
