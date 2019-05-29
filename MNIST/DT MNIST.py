
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split


from sklearn.metrics import mean_squared_error

mnist = fetch_mldata('MNIST original')

X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.25)
depth = []
score_gini = []
score_entropy = []
mse_gini = []
mse_entropy = []


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
depth.append(3)
score =  clf_gini.score(X_test, y_test)
score_gini.append(score)
clf_gini.predict(X_test)
y_true, y_pred = y_test, clf_gini.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_gini.append(mse)
print"=================="

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)

clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)

score =  clf_entropy.score(X_test, y_test)
score_entropy.append(score)
y_true, y_pred = y_test, clf_entropy.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_entropy.append(mse)

print"=================="

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=10, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
depth.append(10)
score =  clf_gini.score(X_test, y_test)
score_gini.append(score)
y_true, y_pred = y_test, clf_gini.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_gini.append(mse)
print"=================="

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=10, min_samples_leaf=5)

clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
score =  clf_entropy.score(X_test, y_test)
score_entropy.append(score)
y_true, y_pred = y_test, clf_entropy.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_entropy.append(mse)

print"=================="
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=50, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
depth.append(50)
score =  clf_gini.score(X_test, y_test)
score_gini.append(score)
y_true, y_pred = y_test, clf_gini.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_gini.append(mse)
print"=================="

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=50, min_samples_leaf=5)

clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
score =  clf_entropy.score(X_test, y_test)
score_entropy.append(score)
y_true, y_pred = y_test, clf_entropy.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_entropy.append(mse)

print"=================="
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=100, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
depth.append(100)
score =  clf_gini.score(X_test, y_test)
score_gini.append(score)
y_true, y_pred = y_test, clf_gini.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_gini.append(mse)
print"=================="

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=100, min_samples_leaf=5)

clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
score =  clf_entropy.score(X_test, y_test)
score_entropy.append(score)
y_true, y_pred = y_test, clf_entropy.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_entropy.append(mse)

print"=================="
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=150, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
depth.append(150)
score =  clf_gini.score(X_test, y_test)
score_gini.append(score)
y_true, y_pred = y_test, clf_gini.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_gini.append(mse)
print"=================="

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=150, min_samples_leaf=5)

clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
score =  clf_entropy.score(X_test, y_test)
score_entropy.append(score)

y_true, y_pred = y_test, clf_entropy.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_entropy.append(mse)
print"=================="
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=200, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
depth.append(200)
score =  clf_gini.score(X_test, y_test)
score_gini.append(score)
y_true, y_pred = y_test, clf_gini.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_gini.append(mse)
print"=================="

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=200, min_samples_leaf=5)

clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
score =  clf_entropy.score(X_test, y_test)
score_entropy.append(score)

y_true, y_pred = y_test, clf_entropy.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_entropy.append(mse)
print"=================="
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=250, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
depth.append(250)
score =  clf_gini.score(X_test, y_test)
score_gini.append(score)
y_true, y_pred = y_test, clf_gini.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_gini.append(mse)
print"=================="

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=250, min_samples_leaf=5)

clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
score =  clf_entropy.score(X_test, y_test)
score_entropy.append(score)

y_true, y_pred = y_test, clf_entropy.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_entropy.append(mse)
print"=================="
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=350, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
depth.append(350)
score =  clf_gini.score(X_test, y_test)
score_gini.append(score)
y_true, y_pred = y_test, clf_gini.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_gini.append(mse)
print"=================="

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=350, min_samples_leaf=5)

clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
score =  clf_entropy.score(X_test, y_test)
score_entropy.append(score)

y_true, y_pred = y_test, clf_entropy.predict(X_test)
mse =  mean_squared_error(y_true, y_pred)
mse_entropy.append(mse)
print"=================="

print "depth", depth
print "score_gini", score_gini
print "score_entropy", score_entropy
print "mse_gini", mse_gini
print "mse_entropy", mse_entropy
