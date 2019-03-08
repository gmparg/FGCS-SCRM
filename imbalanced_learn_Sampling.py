#Sampling techniques to tackle data imbalance

import pandas as pd
import numpy as np

from sklearn import preprocessing as pp
from sklearn import model_selection as ms
from sklearn import svm
from sklearn import tree
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, recall_score, accuracy_score
from sklearn.metrics import average_precision_score as ap_score

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import EasyEnsemble, BalancedBaggingClassifier

import matplotlib.pyplot as plt

X_orig = pd.read_csv("data.csv")    #import CSV dataset and targets
y = pd.read_csv("targets.csv")

Xn = pp.normalize(X_orig)	#normalize input using l2 norm
X = pd.DataFrame(Xn)
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, stratify=y, train_size=0.8)    #train/test split 80/20

y_train = np.ravel(y_train, order='C')	#convert vectors to arrays
y_test = np.ravel(y_test, order='C')

ee = EasyEnsemble(random_state=0, n_subsets=10)
X_train_resampled, y_train_resampled = ee.fit_sample(X_train, y_train)

bbc = BalancedBaggingClassifier(svm.SVC(C=1, gamma=10**5.8),
                                ratio='auto',
                                replacement=False,
                                random_state=0)

svc = svm.SVC()

treeclf = tree.DecisionTreeClassifier(max_depth=6, max_leaf_nodes=13)

#only one sampling technique and one classifier should be uncommented
model = Pipeline([
        ('sampling', RandomOverSampler(random_state=0)),
        #('sampling', RandomUnderSampler(random_state=0)),
        #('sampling', SMOTE()),
        #('sampling', SMOTE(random_state=0, kind='svm')),
        #('sampling', ADASYN()),
        #('sampling', ClusterCentroids(random_state=0)),
        #('sampling', NearMiss(random_state=0, version=3)),
        #('sampling', TomekLinks()),
        #('sampling', OneSidedSelection(random_state=0)),
        #('sampling', SMOTEENN(random_state=0)),
        #('sampling', EasyEnsemble(random_state=0, n_subsets=10)),
        ('clf', clf)
        #('clf', treeclf)
    ])



#code for SVM with grid search, comment out if tree is used
Cs = np.logspace(1, 5, 5)
Gammas = np.logspace(1, 5, 5)
parameters = {'clf__C': Cs, 'clf__gamma': Gammas}
cv = ms.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)  #5-fold cross validation
clf = ms.GridSearchCV(model, parameters, scoring='average_precision', cv=cv, n_jobs=-1, verbose=49) #replace scoring with any metric
clf.fit(grpl_X_train, grpl_y_train)
clf.best_params_    #print best parameters from grid search
clf.best_score_     #print best score from grid search
ap_score(y_test, model.predict(X_test))
f1_score(y_test, model.predict(X_test))
matthews_corrcoef(y_test, model.predict(X_test))
accuracy_score(y_test, clf.predict(X_test))
confusion_matrix(y_test, model.predict(X_test))
recall_score(y_test, model.predict(X_test))

#plot
scores = [x[1] for x in clf.grid_scores_]
scores = np.array(scores).reshape(len(Cs), len(Gammas))

for ind, i in enumerate(Cs):
    plt.plot(Gammas, scores[ind], label='C: ' + str(i))
plt.legend()
plt.xlabel('Gamma')
plt.ylabel('Average precision')
plt.show()


#code for tree, comment out if SVM is used
model.fit(X_train, y_train)
cv = ms.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
ms.cross_val_score(model, X_train, y_train, cv=cv)
model.score(X_test, y_test)
ap_score(y_test, model.predict(X_test))
f1_score(y_test, model.predict(X_test))
matthews_corrcoef(y_test, model.predict(X_test))
accuracy_score(y_test, clf.predict(X_test))
confusion_matrix(y_test, model.predict(X_test))
recall_score(y_test, model.predict(X_test))



