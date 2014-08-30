# by Yi Luo

import numpy as np
from sklearn import ensemble

###############################################################################
# Load data
trainset = np.loadtxt("kaggleTrain.csv", delimiter=',')
sh = trainset.shape
m = sh[0]
X_train, y_train = trainset[:,0:8], trainset[:,11]

testset = np.loadtxt("kaggleTest.csv", delimiter=',')
X_test = testset[:, 0:8]
###############################################################################
# Fit regression model
params = {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

np.savetxt("test_output_registered.csv", y_predict, delimiter=",")