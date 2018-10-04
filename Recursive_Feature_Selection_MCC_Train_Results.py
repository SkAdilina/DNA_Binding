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
#importing training dataset & setting target file
df = pd.read_csv('datasets/train_normalized_32620.csv',low_memory=False)
X = np.asarray(df.iloc[:, 1:-1])
y = np.asarray(df.iloc[:,-1])
train_feature_name = list(df.columns.values[:-1])
print(X.shape)
print(y.shape)

print(X)
print(y.shape)


n_features = 32620



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



for fstt in range(0,n_features,1):
    
    
    forest = RandomForestClassifier(random_state=0) #Initialize with whatever parameters you want to
    forest.fit(X, y)
    
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    x=indices[-1]
    
    if(n_features-fstt == 29823):#the best feature set = 29823
        
        
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
                #print("F{0}:",format(fold))
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evalution
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                mcc.append(matthews_corrcoef(y_true=y_test, y_pred=y_pred))
            
            fold += 1
            
            print ("MCC: {0:3.6}".format(np.mean(mcc)))
        
        
        break
    
    print("{0}".format(train_feature_name[indices[-1]+1]))
    X = np.delete(X, np.s_[x:x+1], axis=1)  #removes the least important feature from training dataset
    train_feature_name = np.delete(train_feature_name, x)

