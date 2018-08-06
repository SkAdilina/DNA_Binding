# total features = 20620 + 12000 = 32620
# total instances = 1075

import pandas as pd
import numpy as np
import csv
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.feature_selection import ColumnSelector 
from sklearn.pipeline import make_pipeline 

# importing the required classifiers
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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold 
 
from datetime import datetime 
from sklearn import datasets 
from datetime import datetime


#importing training dataset & setting target file
df = pd.read_csv('datasets/train_normalized_32620.csv',low_memory=False)
train_data = np.asarray(df.iloc[:, 1:-1])
train_target = np.asarray(df.iloc[:,-1])
train_feature_name = list(df.columns.values[:-1])
print(train_data.shape)
print(train_target.shape)
#print(train_feature_name.shape)

n_features = 32620

start_time = datetime.now()

with open('recursive_feature_selection_for_all_32620_features.csv', 'a+') as csvfile:

    writer = csv.DictWriter(csvfile, fieldnames = ["No", "NumberOfFeatureLeft", "NumberOfFeatureDeleted", "DeletedFeatureName", "TrainAccuracy",])
    writer.writeheader()
    
    for fstt in range(0,n_features,1):

        
        forest = RandomForestClassifier(random_state=0) #Initialize with whatever parameters you want to
        forest.fit(train_data, train_target)

        importances = forest.feature_importances_ #ranked features according to their importance using the Random Forest Classiifer
        indices = np.argsort(importances)[::-1]

        x=indices[-1]

        #Random Forest
        train_acc_score = np.mean(cross_val_score(forest, train_data, train_target,scoring='accuracy',cv=10)) # 10-Fold Cross training accuracy
        
        csvfile.write("{0},{1},{2},{3},".format(fstt+1, n_features-fstt, indices[-1]+1, train_feature_name[indices[-1]+1])) #writes the Current Loop No, Number of Features Left, Number of Features Deleted and the Deleted Feature Name to the csv file.
        csvfile.write("{0:3.6}\n".format(train_acc_score*100)) #writes the TrainAccuracy to the csv file.

        print("{0} --> {1:3.6}".format(train_feature_name[indices[-1]+1], train_acc_score*100))
        train_data = np.delete(train_data, np.s_[x:x+1], axis=1) #removes the least important feature from training dataset
        train_feature_name = np.delete(train_feature_name, x)
 

        end_time = datetime.now()
        print("Duration: {0}\n".format(end_time - start_time))

