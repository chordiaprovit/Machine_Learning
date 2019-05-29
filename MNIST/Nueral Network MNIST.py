import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import classification_report


mnist = fetch_mldata('MNIST original')

X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.25)

print('Got MNIST with %d training- and %d test samples' % (len(y_train), len(y_test)))
print('Digit distribution in whole dataset:', np.bincount(y.astype('int64')))

print('Training model.')
params = {'hidden_layer_sizes': [(256,), (256,256), (128, 256, 128,)]
              }
mlp = MLPClassifier(verbose=10, learning_rate='adaptive')
clf = GridSearchCV(mlp, params, verbose=10, n_jobs=-1, cv=5)
clf.fit(X_train, y_train)
print('Finished with grid search with best mean cross-validated score:', clf.best_score_)
print('Best params appeared to be', clf.best_params_)
# joblib.dump(clf)
clf = clf.best_estimator_

print('Test accuracy:', clf.score(X_test, y_test))

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

# '''
# Output:
# Fetching and loading MNIST data
# Got MNIST with 52500 training- and 17500 test samples
# Digit distribution in whole dataset: [6903 7877 6990 7141 6824 6313 6876 7293 6825 6958]
# Training model.
# Finished with grid search with best mean cross-validated score: 0.97820952381
# Best params appeared to be {'hidden_layer_sizes': (512,)}
# Test accuracy: 0.982057142857

# data = pd.read_csv("/Users/provitchordia/Machine Learning/MNIST data/train.csv")
#
# df_x = data.iloc[:,1:]
# df_y = data.iloc[:,0]
#
# time_taken = []
# iters = []
#
# train_set_loss = []
# overall_accuracy = []
#
#
# x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.30  )
#
# #print collections.Counter(df_y)
#
# nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(100,100*100),
#                  random_state=1,learning_rate_init=0.001,max_iter=200,momentum=0.9)
#
#
# start_time = time.time()
# nn.fit(x_train,y_train)
# end_time = time.time()
#
# pred=nn.predict(x_test)
#
# train_set_score = nn.score(x_train, y_train)
# loss = nn.loss_
# test_set_score = nn.score(x_test, y_test)
#
# time_taken.append(end_time - start_time)
#
#
#
#
# # a=y_test.values
#
#
# #print collections.Counter(pred)
#
# # correct_class=0.0
# # miss_class = 0.0
# #
# # for i in range(len(pred)):
# #     if pred[i]==a[i]:
# #         correct_class = correct_class +1
# #
# # overall_accuracy.append(correct_class/len(pred))
# print "--------"
#
# # cm = confusion_matrix(y_test, pred)
# #
# # plt.matshow(cm)
# # plt.title('Accuracy in prediction')
# # plt.colorbar()
# # plt.ylabel('True label')
# # plt.xlabel('Predicted label')
# # plt.show()
#
# print "train_set_score", train_set_score
# print "test_set_score", test_set_score
# print "loss", loss
#
# print "time_taken", time_taken

