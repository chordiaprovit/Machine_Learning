import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt




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
            
            print("Start fitting. This may take a while")
            examples = len(data['train']['X'])
            clf.fit(data['train']['X'][:examples], data['train']['y'][:examples])
            score = clf.score(data['train']['X'][:examples], data['train']['y'][:examples])
            
            predicted = clf.predict(data['test']['X'])
            scoretest = metrics.accuracy_score(data['test']['y'], predicted)
            print("Accuracy rbf: %0.4f" % scoretest)
            # print (clf.get_params(deep=True))
            
            test_set_score.append(scoretest)
            train_set_score.append(score)
        
        print "C:", Cs
        print "Gammas:", gammas
        print "train_set_score", train_set_score
        print "test_set_score", test_set_score
        print "--------"

    line1, = plt.plot(gammas, train_set_score, label='training set score')
    line2 = plt.plot(gammas, test_set_score, label='test set score')
   
    plt.title('Accuracy at diffrent gammas')
    plt.xlabel("gamma")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    
    index = test_set_score.index(max(test_set_score))
    print index
    best_gamma = gammas[index]
    best_C = Cs[index]

    svm1 = SVC(probability=False, kernel="rbf", C=best_C, gamma=best_gamma)
    svm1.fit(data['train']['X'][:examples], data['train']['y'][:examples])
    pred1 = svm1.predict(data['test']['X'])
    cm = confusion_matrix(data['test']['y'], pred1)
    print(cm)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



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

    # clf = SVC(probability=False,  # cache_size=200,
    #          kernel="rbf", C=2.8, gamma=.0073)
    
    learnerrbf(data)


def get_data():
    from sklearn.datasets import fetch_mldata
    from sklearn.utils import shuffle
    import pandas as pd
    from numpy.random import RandomState
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv("/Users/provitchordia/Machine Learning/Adult.csv")
    print df.shape
    df.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]

    # create a numpy.random.RandomState so that we can reproduce the same results each time
    rs = RandomState(130917)

    # convert the <=50Ks into -1 and the >50K into +1
    df["Income"] = df["Income"].map({" <=50K": -1, " >50K": 1})

    # extract the response variable into a numpy array and drop it from the dataframe:
    y_all = df["Income"].values
    df.drop("Income", axis=1, inplace=True, )

    # print df.CapitalGain.value_counts()
    # print df.CapitalLoss.value_counts()

    # more than 95% of CapitalGain and CapitalLoss column consists of 0. So dropping the column
    df.drop("CapitalGain", axis=1, inplace=True, )
    df.drop("CapitalLoss", axis=1, inplace=True, )

    # Convert the Age, fnlwgt, EducationNum and HoursPerWeek to floating point
    df.Age = df.Age.astype(float)
    df.fnlwgt = df.fnlwgt.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.HoursPerWeek = df.HoursPerWeek.astype(float)

    # one-hot encoding to transform them into numerical features
    df = pd.get_dummies(df, columns=[
        "WorkClass", "Education", "MaritalStatus", "occupation", "Relationship",
        "Race", "Gender", "NativeCountry",
    ])

    # check if data is imbalanced
    pd.value_counts(pd.Series(y_all))

    print df[:0]

    # Since data is imbalanced, using stratify split to ensure data is splitted evenly in test and train sets
    X_train, X_test, y_train, y_test = train_test_split(
        df, y_all, test_size=0.30, stratify=y_all, random_state=rs, )
    data = {'train': {'X': X_train,
                      'y': y_train},
            'test': {'X': X_test,
                     'y': y_test}}
    return data


if __name__ == '__main__':
    main()