from __future__ import division

# By pass warnings
#====================================================================
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

# Define Important Library
#====================================================================
import pandas as pd
import numpy as np

# Start from 1 always, no random state
#====================================================================
np.random.seed(1)

# scikit-learn library import
#====================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import csv
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline

from sklearn import svm
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold

from datetime import datetime
from sklearn import datasets


# Load the Featureset:
#====================================================================

#importing training dataset & setting target file
df = pd.read_csv('datasets/train_G_normalized_nearest_neigbour.csv',low_memory=False)
train1_data = np.asarray(df.iloc[:, 1:-1])
train1_target = np.asarray(df.iloc[:,-1])
train1_feature_name = list(df.columns.values[:-1])
print(train1_data.shape)
#print(train1_feature_name)

df = pd.read_csv('datasets/train_C_normalized_gap.csv',low_memory=False)
train2_data = np.asarray(df.iloc[:, 1:-1])
train2_target = np.asarray(df.iloc[:,-1])
train2_feature_name = list(df.columns.values[1:-1])
print(train2_data.shape)
#print(train2_feature_name)

train3_data = np.hstack((train1_data,train2_data))
train3_feature_name = np.hstack((train1_feature_name,train2_feature_name))
print(train3_data.shape)

df = pd.read_csv('datasets/train_E_normalized_monogram.csv',low_memory=False)
train4_data = np.asarray(df.iloc[:, 1:-1])
train4_target = np.asarray(df.iloc[:,-1])
train4_feature_name = list(df.columns.values[1:-1])
print(train4_data.shape)

df = pd.read_csv('datasets/train_F_normalized_bigram.csv',low_memory=False)
train5_data = np.asarray(df.iloc[:, 1:-1])
train5_target = np.asarray(df.iloc[:,-1])
train5_feature_name = list(df.columns.values[1:])
print(train4_data.shape)

train6_data = np.hstack((train4_data,train5_data))
train6_feature_name = np.hstack((train4_feature_name,train5_feature_name))
print(train6_data.shape)

X = np.hstack((train3_data,train6_data))
train_feature_name = np.hstack((train3_feature_name,train6_feature_name))
y = train1_target
print(X)
print(y.shape)


print("GCEF\n");



# Define classifiers within a list
#====================================================================
Classifiers = [
               RandomForestClassifier(random_state=0),
               ExtraTreesClassifier(random_state=0),
               svm.SVC(kernel='linear', C=1, probability=True),
               LogisticRegression(C=1),
               AdaBoostClassifier(random_state=0),
               DecisionTreeClassifier(),
               GaussianNB(),
               KNeighborsClassifier(),
               LinearDiscriminantAnalysis(n_components=500),
               ]

# Spliting with 10-Folds :
#====================================================================
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=False)


# Pick all classifier within the Classifier list and test one by one
#====================================================================
for classifier in Classifiers:
    mcc = []
    fold = 1
    print('____________________________________________')
    print('Classifier: '+classifier.__class__.__name__)
    model = classifier
    for train_index, test_index in cv.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        
        y_train = y[train_index]
        y_test = y[test_index]
        
        # print the fold number and numbber of feature after selection
        # -----------------------------------------------------------------
        #print("F{0}:",format(fold))
        
        # Train model
        # -----------------------------------------------------------------
        model.fit(X_train, y_train)
        
        # Evalution
        # -----------------------------------------------------------------
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        
        mcc.append(matthews_corrcoef(y_true=y_test, y_pred=y_pred))
        
        fold += 1
    
    print ("MCC: {0:3.6}".format(np.mean(mcc)))


