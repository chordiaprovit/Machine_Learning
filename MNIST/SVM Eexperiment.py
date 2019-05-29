import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit



# Loading the Digits dataset
mnist = fetch_mldata('MNIST original')
X, y = mnist.data, mnist.target
print X.shape
print y.shape

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
# n_samples = len(mnist)
# print n_samples
# X = mnist(n_samples, -1)
# y = mnist.target


# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

clf_ = SVC(kernel='rbf')
Cs = [1, 10, 100, 1000]
Gammas = [1e-3, 1e-4]
clf = GridSearchCV(clf_,
            dict(C=Cs,
                 gamma=Gammas),
                 cv=2,
                 pre_dispatch='1*n_jobs',
                 n_jobs=1)

clf.fit(X, y)

scores = [x[1] for x in clf.grid_scores_]
scores = np.array(scores).reshape(len(Cs), len(Gammas))
print "score", scores


for ind, i in enumerate(Cs):
    plt.plot(Gammas, scores[ind], label='C: ' + str(i))
plt.legend()
plt.xlabel('Gamma')
plt.ylabel('Mean score')
plt.show()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()




