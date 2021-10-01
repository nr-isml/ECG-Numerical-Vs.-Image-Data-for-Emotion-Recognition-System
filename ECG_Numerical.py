# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 19:38:40 2020

@author: nr.isml
"""

#import libraries
import pandas as pd
#READ CSV FILE
data = pd.read_csv('<insert input file>')
features = data.drop(['Arousal', 'Valence'], axis=1)
target = data['Valence']

#SPLIT DATA INTO TRAIN AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features, target,test_size=0.20,random_state=42,stratify=target)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#PREPROCESSING PART - STANDARD SCALER TO STANDARDIZE DATA
from sklearn import preprocessing
scaler = preprocessing.StandardScaler() 
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)

#DIMENSIONALITY REDUCTION USING LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

#CREATE CLASSIFIER MODEL
from sklearn.svm import SVC
model = SVC(random_state=42)
parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1 , 1, 10], 'gamma': [0.1, 1, 10]}

#SPECIFY CROSS VALIDATION
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, shuffle=True, random_state=42)

#PARAMETER TUNED USING GRIDSEARCHCV
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(model, param_grid=parameters, scoring='accuracy', n_jobs=1, cv=cv, refit=True)
clf.fit(X_train, y_train)
print()
print('Best parameters ', clf.best_params_)
print()
print('------------THIS IS TRAINING RESULT------------')
print()
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf.best_estimator_, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Acc: %0.2f' % (scores.max() * 100))

from sklearn.metrics import accuracy_score
print()
print('------------THIS IS TESTING RESULT------------')
print()
print('Acc: %0.2f' % (accuracy_score(y_test,clf.best_estimator_.predict(X_test)) * 100))


















