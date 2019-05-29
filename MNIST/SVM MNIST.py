import numpy as np
from sklearn.svm import SVC
from sklearn import metrics

def learnerrbf(data):

    C_value = np.arange(1, 5, 1)
    Gamma = [0.001, 0.005, 0.0075, 0.01]


    for C in C_value:
        Cs = []
        gammas = []
        train_set_score = []
        test_set_score = []
        
        for G in Gamma:
            Cs.append(C)
            gammas.append(G)
            
            clf = SVC(probability=False, kernel="rbf", C=C, gamma=G)
            # print("Start fitting. This may take a while")
            examples = len(data['train']['X'])
            clf.fit(data['train']['X'][:examples], data['train']['y'][:examples])
            score = clf.score(data['train']['X'][:examples], data['train']['y'][:examples])
            
            predicted = clf.predict(data['test']['X'])
            scoretest = metrics.accuracy_score(data['test']['y'], predicted)
            # print("Accuracy rbf: %0.4f" % scoretest)
            # print (clf.get_params(deep=True))
            
            test_set_score.append(scoretest)
            train_set_score.append(score)
            
        print "C:", Cs
        print "Gammas:", gammas
        print "train_set_score", train_set_score
        print "test_set_score", test_set_score
        print "--------"
        

# def learnerlin(data, parmC, parmG):
#     clf = SVC(probability=False, kernel="linear", C=parmC, gamma=parmG)
#     print("Start fitting. This may take a while")
#     examples = len(data['train']['X'])
#     clf.fit(data['train']['X'][:examples], data['train']['y'][:examples])
#
#     predicted = clf.predict(data['test']['X'])
#     print "C", parmC
#     print "gamma", parmG
#     print("Accuracy lin: %0.4f" % metrics.accuracy_score(data['test']['y'], predicted))
#     print (clf.get_params(deep=True))
#     print "--------"
#
def main():
    data = get_data()

    #clf = SVC(probability=False,  # cache_size=200,
    #          kernel="rbf", C=2.8, gamma=.0073)

    learnerrbf(data)


def get_data():
    from sklearn.datasets import fetch_mldata
    from sklearn.utils import shuffle
    mnist = fetch_mldata('MNIST original')

    X = mnist.data
    y = mnist.target

    filter = np.where((y == 8) | (y == 9))

    X = X[filter]
    y = y[filter]

    # Scale data to [-1, 1] - This is of mayor importance!!!
    x = X/255.0*2 - 1

    x, y = shuffle(x, y, random_state=0)

    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.33,
                                                            random_state=42)
    data = {'train': {'X': x_train,
                          'y': y_train},
                'test': {'X': x_test,
                         'y': y_test}}
    return data


if __name__ == '__main__':
    main()