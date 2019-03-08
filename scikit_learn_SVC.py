#Support Vector Machine Classification

import pandas as pd
import numpy as np

from sklearn import preprocessing as pp
from sklearn import model_selection as ms
from sklearn import svm
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, recall_score, accuracy_score
from sklearn.metrics import average_precision_score as ap_score

from sklearn.externals import joblib

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

X_orig = pd.read_csv("data.csv")    #import CSV dataset and targets
y = pd.read_csv("targets.csv")

Xn = pp.normalize(X_orig)	#normalize input using l2 norm
X = pd.DataFrame(Xn)
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, stratify=y, train_size=0.8)    #train/test split 80/20

y_train = np.ravel(y_train, order='C')	#convert vectors to arrays
y_test = np.ravel(y_test, order='C')

svc = svm.SVC()

Cs = np.logspace(1, 5, 5)       #set parameters for grid search
Gammas = np.logspace(1, 5, 5)
parameters = {'C': Cs, 'gamma': Gammas}
cv = ms.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)  #5-fold cross validation
clf = ms.GridSearchCV(svc, parameters, scoring='average_precision', cv=cv, n_jobs=-1, verbose=49) #replace scoring with any metric

clf.fit(X_train, y_train)

clf.best_params_    #print best parameters from grid search
clf.best_score_     #print best score from grid search
clf.score(X_test, y_test)
ap_score(y_test, clf.predict(X_test))
f1_score(y_test, clf.predict(X_test))
matthews_corrcoef(y_test, clf.predict(X_test))
accuracy_score(y_test, clf.predict(X_test))
recall_score(y_test, clf.predict(X_test))
confusion_matrix(y_test, clf.predict(X_test))


#model persistence
joblib.dump(clf, 'svc.joblib') 


#plot

scores = [x[1] for x in clf.grid_scores_]
scores = np.array(scores).reshape(len(Cs), len(Gammas))

for ind, i in enumerate(Cs):
    plt.plot(Gammas, scores[ind], label='C: ' + str(i))
plt.legend()
plt.xlabel('Gamma')
plt.ylabel('Average precision')
plt.show()

#heat map

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

scores = clf.cv_results_['mean_test_score'].reshape(len(Cs), len(Gammas))
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(Gammas)), Gammas, rotation=45)
plt.yticks(np.arange(len(Cs)), Cs)
plt.title('Test Score: AP')
plt.show()
