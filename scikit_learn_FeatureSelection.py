#Feature Selection

import pandas as pd
import numpy as np

from sklearn import preprocessing as pp
from sklearn import model_selection as ms

from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif, RFECV, SelectFromModel

from sklearn import svm

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import 

from sklearn.decomposition import PCA

import seaborn

import matplotlib.pyplot as plt


X_orig = pd.read_csv("data.csv")    #import CSV dataset and targets
y = pd.read_csv("targets.csv")

Xn = pp.normalize(X_orig)	#normalize input using l2 norm
X = pd.DataFrame(Xn)

#remove all low-variance features
vt_selector = VarianceThreshold(threshold=0.01)
vt_res = vt_selector.fit_transform(X)
vt_selector.get_support()

#select the k best features using f_classif metric
kb_f_selector = SelectKBest(k=32) 

#x-squares requires features to be strictly positive
scaler = pp.MinMaxScaler()
X_x2 = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

#select the k best features using x-squared metric
kb_x2_selector = SelectKBest(chi2, k=32)
kb_x2_res = kb_x2_selector.fit_transform(X_x2, y)

#select the k best features using mutual information metric
kb_mi_selector = SelectKBest(mutual_info_classif, k=32) 
kb_mi_res = selector.fit_transform(X, y)


#recursive feature elimination
estimator = svm.SVC(kernel="linear")
cv = ms.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
rfecv_selector = RFECV(estimator, step=1, cv=cv, scoring='average_precision', verbose=49, n_jobs=-1)
rfecv_selector = selector.fit(X, y)
rfecv_selector.support_

#plots: replace vt_selector with desired selector

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(vt_selector.grid_scores_) + 1), vt_selector.grid_scores_)
plt.show()


#using ExtraTrees for feature selection
clf = ExtraTreesClassifier(n_estimators=250, verbose=49)
clf = clf.fit(X, y)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
            axis=0)
indices = np.argsort(importances)[::-1]

#Plot feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


#PCA for feature importance
pca = PCA()
pca_data = pca.fit_transform(X)
seaborn.heatmap(np.log(pca.inverse_transform(np.eye(X.shape[1]))), cmap="hot", cbar=False)
plt.show()

