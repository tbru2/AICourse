import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

data = np.loadtxt('input3.csv', delimiter=',')
x_train, x_test, y_train, y_test = train_test_split(
    data[:,0:2],data[:,2], test_size=.4, random_state=0 
)

clf_linear = svm.SVC(kernel='linear', C=100)
clf_linear.fit(x_train, y_train)
print "svm_linear"
print(max(cross_val_score(clf_linear,x_train,y_train,cv=5)))
print(max(cross_val_score(clf_linear,x_test,y_test,cv=5)))

param_grid_poly = {'C' : [0.1, 1, 3], 'degree' : [4, 5, 6], 'gamma' : [0.1, 0.5]}

clf_poly = svm.SVC(kernel='poly')
grid_search_poly = GridSearchCV(clf_poly, param_grid_poly, cv=5)
grid_search_poly.fit(x_train, y_train)
print "poly"
print (grid_search_poly.best_score_)
pred_poly = grid_search_poly.predict(x_test)
print accuracy_score(y_test, pred_poly)

param_grid_rbf = {'C' : [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma' : [0.1, 0.5, 1, 3, 6, 10]}
clf_rbf = svm.SVC(kernel='rbf')
grid_search_rbf = GridSearchCV(clf_rbf, param_grid_rbf, cv=5)
grid_search_rbf.fit(x_train, y_train)
print "rbf"
print grid_search_rbf.best_score_
pred_rbf = grid_search_rbf.predict(x_test)
print accuracy_score(y_test, pred_rbf)

param_grid_log = {'C' : [0.1, 0.5, 1, 5, 10, 50, 100]}
clf_log = LogisticRegression()
grid_search_log = GridSearchCV(clf_log, param_grid_log, cv=5)
grid_search_log.fit(x_train,y_train)
print "log"
print grid_search_log.best_score_
pred_log = grid_search_log.predict(x_test)
print accuracy_score(y_test, pred_log)

param_grid_neighbors = {'n_neighbors' : [ i for i in range(1,51)], 'leaf_size' : [i for i in range(5,61,5)] }
clf_neighbors = KNeighborsClassifier()
grid_search_neighbors = GridSearchCV(clf_neighbors, param_grid_neighbors, cv=5)
grid_search_neighbors.fit(x_train, y_train)
print "neigbors"
print grid_search_neighbors.best_score_
pred_neighbors = grid_search_neighbors.predict(x_test)
print accuracy_score(y_test, pred_neighbors)

param_grid_decTree = {'max_depth' : [i for i in range(1,51)], 'min_samples_split' : [i for i in range(2,11)]}
clf_decTree = DecisionTreeClassifier(random_state=0)
grid_search_decTree = GridSearchCV(clf_decTree, param_grid_decTree, cv=5)
grid_search_decTree.fit(x_train, y_train)
print "decTree"
print grid_search_decTree.best_score_
pred_decTree = grid_search_decTree.predict(x_test)
print accuracy_score(y_test, pred_decTree)

param_grid_rand_for = {'max_depth' : [i for i in range(1,51)], 'min_samples_split' : [i for i in range(2,11)]}
clf_rand = RandomForestClassifier()
grid_search_rand = GridSearchCV(clf_rand, param_grid_rand_for, cv=5)
grid_search_rand.fit(x_train, y_train)
print "random forest"
print grid_search_rand.best_score_
pred_rand = grid_search_rand.predict(x_test)
print accuracy_score(y_test, pred_rand)