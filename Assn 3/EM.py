import numpy as np
import itertools

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

    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
    for n in n_components]

    plt.plot(n_components, [m.bic(data) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(data) for m in models], label='AIC')
    plt.title('likelihood predicted by a GMM for wine')
    plt.axis('tight')
    plt.xlabel('components ')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def adult():
    # Number of samples per component
    data = getAdultData()

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
    for n in n_components]

    plt.plot(n_components, [m.bic(data) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(data) for m in models], label='AIC')
    plt.title('likelihood predicted by a GMM for Adult')
    plt.axis('tight')
    plt.xlabel('components ')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    wine()
    adult()