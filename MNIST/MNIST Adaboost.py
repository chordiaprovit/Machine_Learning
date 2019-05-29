from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.25)

def learner(parm1, parm2):
    clf = AdaBoostClassifier(n_estimators=parm1,
                                 algorithm='SAMME.R',
                                 base_estimator=DecisionTreeClassifier(criterion='entropy', min_samples_split=parm2))
    clf.fit(X_train, y_train)
    label_predict = clf.predict(X_test)
    
    precision = precision_score(y_test, label_predict, average='weighted')
    
    from sklearn.metrics import recall_score
    recall = recall_score(y_test, label_predict, average='weighted')
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, label_predict)
    print "accuracy", accuracy
    # print('estimator size:', estimator_size)
    print "estimator", parm1
    print "sample split", parm2
    print(str(precision)+'\t'+ str(recall)+'\t' +str(f1)+'\t'+ str(accuracy))
    print "---"



learner(20,5)
learner(30,5)
learner(50,5)
learner(100,5)

learner(20,10)
learner(30,10)
learner(50,10)
learner(100,10)


learner(20,15)
learner(30,15)
learner(50,15)
learner(100,15)