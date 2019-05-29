
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import cross_validation

from sklearn.metrics import mean_squared_error

mnist = fetch_mldata('MNIST original')

X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.25)
depth = []
leaf = []
score_gini = []
score_entropy = []
mse_gini = []
mse_entropy = []

def gini(parms):
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=parms, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    score =  clf_gini.score(X_test, y_test, )
    score_gini.append(score)
    clf_gini.predict(X_test)
    y_true, y_pred = y_test, clf_gini.predict(X_test)
    mse =  mean_squared_error(y_true, y_pred)
    mse_gini.append(mse)
    return

def entropy(parms):
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                         max_depth=parms, min_samples_leaf=5)
    
    clf_entropy.fit(X_train, y_train)
    score = clf_entropy.score(X_test, y_test)
    clf_entropy.predict(X_test)
    score_entropy.append(score)
    
    y_true, y_pred = y_test, clf_entropy.predict(X_test)
    mse = mean_squared_error(y_true, y_pred)
    mse_entropy.append(mse)
    return


def gini_leaf(parms):
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100,
                                      max_depth=50, min_samples_leaf=parms)
    clf_gini.fit(X_train, y_train)
    score = clf_gini.score(X_test, y_test)
    score_gini.append(score)
    
    clf_gini.predict(X_test)
    y_true, y_pred = y_test, clf_gini.predict(X_test)
    mse = mean_squared_error(y_true, y_pred)
    mse_gini.append(mse)
    return


def entropy_leaf(parms):
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                         max_depth=50, min_samples_leaf=parms)
    
    clf_entropy.fit(X_train, y_train)
    score = clf_entropy.score(X_test, y_test)
    clf_entropy.predict(X_test)
    score_entropy.append(score)
    
    y_true, y_pred = y_test, clf_entropy.predict(X_test)
    mse = mean_squared_error(y_true, y_pred)
    mse_entropy.append(mse)
    return

def call_depth(parm):
    depth.append(parm)
    gini(parm)
    entropy(parm)
    return

def call_leaf(parm):
    leaf.append(parm)
    gini_leaf(parm)
    entropy_leaf(parm)
    return
    

call_depth(10)
call_depth(20)
call_depth(30)
call_depth(40)
call_depth(50)
call_depth(60)
call_depth(70)

print "depth", depth
print "score_gini", score_gini
print "score_entropy", score_entropy
print "mse_gini", mse_gini
print "mse_entropy", mse_entropy
print "----------------"

a= plt.plot(depth, score_gini, label='gini score')
a= plt.plot(depth, score_entropy, label='entropy score')


plt.title('Score and MSE at diffrent tree depth(min leaf size = 5)')
plt.xlabel("depth")
plt.ylabel("score/MSE")
plt.legend()
plt.show()


depth = []
leaf = []
score_gini = []
score_entropy = []
mse_gini = []
mse_entropy = []

call_leaf(5)
call_leaf(10)
call_leaf(20)
call_leaf(30)
call_leaf(50)
call_leaf(75)
call_leaf(100)

print "leaf", leaf
print "score_gini", score_gini
print "score_entropy", score_entropy
print "mse_gini", mse_gini
print "mse_entropy", mse_entropy
print "----------------"

plt.plot(leaf, score_gini, label='gini score')
score_gini[:] = [x * 2 for x in score_gini]
plt.plot(leaf, score_entropy, label='entropy score')
score_entropy[:] = [x * 2 for x in score_entropy]
plt.plot(leaf, mse_gini, label='gini MSE')
plt.plot(leaf, mse_entropy, label='entropy MSE')

plt.title('Score and MSE at diffrent leaf size(max_depth = 50)')
plt.xlabel("leaf size")
plt.ylabel("score*2 / MSE")
plt.legend()
plt.show(b)



