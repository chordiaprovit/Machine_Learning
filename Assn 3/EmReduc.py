import numpy as np
import itertools
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.mixture import GaussianMixture
from util import getWineData,getAdultData

def wine():
    # Number of samples per component
    data = getWineData()

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    name = "wine"
    measure(data,name)


def adult():
    # Number of samples per component
    data = getAdultData()

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    name = "Adult"
    measure(data,name)

def measure(X,name):
    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
    for n in n_components]

    plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
    plt.title('likelihood predicted by a GMM for ' + name)
    plt.axis('tight')
    plt.xlabel('components ')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    wine()
    adult()
    pcData = pd.read_csv("/Users/provitchordia/Machine Learning/Assn 3/output file/wine_pca.csv", skiprows=1,
                         usecols=range(1, 8))
    name = "wine PCA"
    measure(pcData, name)

    rfData = pd.read_csv("/Users/provitchordia/Machine Learning/Assn 3/output file/wine_rf.csv", skiprows=1,
                         usecols=range(1, 4))
    name = "wine Random forest"
    measure(rfData, name)

    rpData = pd.read_csv("/Users/provitchordia/Machine Learning/Assn 3/output file/wine_rp.csv", skiprows=1,
                         usecols=range(1, 7))
    name = "wine Random Projection"
    measure(rpData, name)

    icaData = pd.read_csv("/Users/provitchordia/Machine Learning/Assn 3/output file/wine_ica.csv", skiprows=1,
                          usecols=range(1, 7))
    name = "wine ICA"
    measure(icaData, name)