#Decision Tree Classification

import pandas as pd
import numpy as np

from sklearn import preprocessing as pp
from sklearn import model_selection as ms
from sklearn import tree
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, recall_score, accuracy_score
from sklearn.metrics import average_precision_score as ap_score

import graphviz

X_orig = pd.read_csv("data.csv")    #import CSV dataset and targets
y = pd.read_csv("targets.csv")

Xn = pp.normalize(X_orig)	#normalize input using l2 norm
X = pd.DataFrame(Xn)
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, stratify=y, train_size=0.8)    #train/test split 80/20

y_train = np.ravel(y_train, order='C')	#convert vectors to arrays
y_test = np.ravel(y_test, order='C')


clf = tree.DecisionTreeClassifier(max_depth=6, max_leaf_nodes=13)   #limiting tree size
clf.fit(X_train, y_train)
cv = ms.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
ms.cross_val_score(clf, X_train, y_train, cv=cv)
clf.score(X_test, y_test)
ap_score(y_test, clf.predict(X_test))
matthews_corrcoef(y_test, clf.predict(X_test))
f1_score(y_test, clf.predict(X_test))
confusion_matrix(y_test, clf.predict(X_test))
recall_score(y_test, clf.predict(X_test))
clf.tree_.max_depth
clf.tree_.node_count


#tree visualisation
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("tree6-13")

