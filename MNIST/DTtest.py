import sklearn.model_selection as ms
import pandas as pd
from helpers import basicResults, dtclf_pruned, makeTimingCurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

def DTpruningVSnodes(clf, alphas, trgX, trgY, dataset):
    '''Dump table of pruning alpha vs. # of internal nodes'''
    out = {}
    for a in alphas:
        clf.set_params(**{'DT__alpha': a})
        clf.fit(trgX, trgY)
        out[a] = clf.steps[-1][-1].numNodes()
        print(dataset, a)
    out = pd.Series(out)
    out.index.name = 'alpha'
    out.name = 'Number of Internal Nodes'
    out.to_csv('./output/DT_{}_nodecounts.csv'.format(dataset))
    
    return


# Load Data
mnist = fetch_mldata('MNIST original')

X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.25)




# Search for good alphas
alphas = [-1, -1e-3, -(1e-3) * 10 ** -0.5, -1e-2, -(1e-2) * 10 ** -0.5, -1e-1, -(1e-1) * 10 ** -0.5, 0,
          (1e-1) * 10 ** -0.5, 1e-1, (1e-2) * 10 ** -0.5, 1e-2, (1e-3) * 10 ** -0.5, 1e-3]
# alphas=[0]
pipeM = Pipeline([('Scale', StandardScaler()),
                  ('Cull1', SelectFromModel(RandomForestClassifier(random_state=1), threshold='median')),
                  ('Cull2', SelectFromModel(RandomForestClassifier(random_state=2), threshold='median')),
                  ('Cull3', SelectFromModel(RandomForestClassifier(random_state=3), threshold='median')),
                  ('Cull4', SelectFromModel(RandomForestClassifier(random_state=4), threshold='median')),
                  ('DT', dtclf_pruned(random_state=55))])

pipeA = Pipeline([('Scale', StandardScaler()),
                  ('DT', dtclf_pruned(random_state=55))])

params = {'DT__criterion': ['gini', 'entropy'], 'DT__alpha': alphas, 'DT__class_weight': ['balanced']}

MNIST_clf = basicResults(pipeM, X_train, y_train, X_test, y_test, params, 'DT', 'MNIST')


# madelon_final_params = {'DT__alpha': -0.00031622776601683794, 'DT__class_weight': 'balanced', 'DT__criterion': 'entropy'}
# adult_final_params = {'class_weight': 'balanced', 'alpha': 0.0031622776601683794, 'criterion': 'entropy'}
MNIST_final_params = MNIST_clf.best_params_


pipeM.set_params(**MNIST_final_params)
makeTimingCurve(X, y, pipeM, 'DT', 'MNIST')


DTpruningVSnodes(pipeM, alphas, X_train, y_train, 'MNIST')
