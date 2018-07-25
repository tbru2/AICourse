import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = np.loadtxt('input3.csv', delimiter=',')
x_train, x_test, y_train, y_test = train_test_split(
    data[:,0:2],data[:,2], test_size=.4, random_state=0 
)

param_grid_rand_for = {'max_depth' : [i for i in range(1,51)], 'min_samples_split' : [i for i in range(2,11)]}
clf_rand = RandomForestClassifier()
grid_search_rand = GridSearchCV(clf_rand, param_grid_rand_for, cv=5)
grid_search_rand.fit(x_train, y_train)
print "random forest"
print grid_search_rand.best_score_
pred_rand = grid_search_rand.predict(x_test)
print accuracy_score(y_test, pred_rand)