import pandas as pd
import numpy as np
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


#importing training dataset & setting target file
df = pd.read_csv('datasets/train_G_normalized_nearest_neigbour.csv',low_memory=False)
train1_data = np.asarray(df.iloc[:, 1:-1])
train1_target = np.asarray(df.iloc[:,-1])
train1_feature_name = list(df.columns.values[:-1])

df = pd.read_csv('datasets/train_C_normalized_gap.csv',low_memory=False)
train2_data = np.asarray(df.iloc[:, 1:-1])
train2_target = np.asarray(df.iloc[:,-1])
train2_feature_name = list(df.columns.values[1:-1])

train3_data = np.hstack((train1_data,train2_data))
train3_feature_name = np.hstack((train1_feature_name,train2_feature_name))

df = pd.read_csv('datasets/train_E_normalized_monogram.csv',low_memory=False)
train4_data = np.asarray(df.iloc[:, 1:-1])
train4_target = np.asarray(df.iloc[:,-1])
train4_feature_name = list(df.columns.values[1:-1])

df = pd.read_csv('datasets/train_F_normalized_bigram.csv',low_memory=False)
train5_data = np.asarray(df.iloc[:, 1:-1])
train5_target = np.asarray(df.iloc[:,-1])
train5_feature_name = list(df.columns.values[1:])

train6_data = np.hstack((train4_data,train5_data))
train6_feature_name = np.hstack((train4_feature_name,train5_feature_name))

train_data = np.hstack((train3_data,train6_data))
train_feature_name = np.hstack((train3_feature_name,train6_feature_name))
train_target = train1_target
print(train_data)
print(train_target.shape)




#importing testing dataset & setting target file
df = pd.read_csv('datasets/test_G_normalized_nearest_neigbour.csv',low_memory=False)
test1_data = np.asarray(df.iloc[:, 1:-1])
test1_target = np.asarray(df.iloc[:,-1])
test1_feature_name = list(df.columns.values[:-1])

df = pd.read_csv('datasets/test_C_normalized_gap.csv',low_memory=False)
test2_data = np.asarray(df.iloc[:, 1:-1])
test2_target = np.asarray(df.iloc[:,-1])
test2_feature_name = list(df.columns.values[1:-1])

test3_data = np.hstack((test1_data,test2_data))
test3_feature_name = np.hstack((test1_feature_name,test2_feature_name))

df = pd.read_csv('datasets/test_E_normalized_monogram.csv',low_memory=False)
test4_data = np.asarray(df.iloc[:, 1:-1])
test4_target = np.asarray(df.iloc[:,-1])
test4_feature_name = list(df.columns.values[1:-1])

df = pd.read_csv('datasets/test_F_normalized_bigram.csv',low_memory=False)
test5_data = np.asarray(df.iloc[:, 1:-1])
test5_target = np.asarray(df.iloc[:,-1])
test5_feature_name = list(df.columns.values[1:])

test6_data = np.hstack((test4_data,test5_data))
test6_feature_name = np.hstack((test4_feature_name,test5_feature_name))

test_data = np.hstack((test3_data,test6_data))
test_feature_name = np.hstack((test3_feature_name,test6_feature_name))
test_target = test1_target
print(test_data.shape)
print(test_target.shape)

print("GCEF\n");



#Random Forest
forest = RandomForestClassifier(random_state=0) #Initialize with whatever parameters you want to
train_acc_score = np.mean(cross_val_score(forest, train_data, train_target,scoring='accuracy',cv=10)) # 10-Fold Cross validation
train_recall_score = np.mean(cross_val_score(forest, train_data, train_target,scoring='recall',cv=10))
train_specificity_score = (2*train_acc_score) - train_recall_score
train_auroc_score = np.mean(cross_val_score(forest, train_data, train_target,scoring='roc_auc',cv=10))
train_auPR_score = np.mean(cross_val_score(forest, train_data, train_target,scoring='average_precision',cv=10))

forest.fit(train_data, train_target)
test_rf = forest.predict(test_data)
test_acc_score_rf =accuracy_score(test_target, test_rf)
test_recall_score_rf = recall_score(test_target, test_rf, average=None) #sensitivtiy
test_specificity_rf = (2*test_acc_score_rf) - test_recall_score_rf #specificity or confusion matrix
test_auroc_score_rf = roc_auc_score(test_target, test_rf)
test_auPR_score_rf = average_precision_score(test_target, test_rf)
test_mcc_score_rf = matthews_corrcoef(test_target, test_rf)

print("Random Forest,{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},".format(train_acc_score*100, train_recall_score, train_specificity_score, train_auroc_score, train_auPR_score))
print("{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},{5:3.6}\n".format(test_acc_score_rf*100, np.mean(test_recall_score_rf), np.mean(test_specificity_rf), test_auroc_score_rf, test_auPR_score_rf,test_mcc_score_rf))

#ExtraTrees
ExT = ExtraTreesClassifier(random_state=0)
train_acc_score_ext = np.mean(cross_val_score(ExT, train_data, train_target,scoring='accuracy',cv=10))
train_recall_score_ext = np.mean(cross_val_score(ExT, train_data, train_target,scoring='recall',cv=10))
train_specificity_score_ext = (2*train_acc_score_ext) - train_recall_score_ext
train_auroc_score_ext = np.mean(cross_val_score(ExT, train_data, train_target,scoring='roc_auc',cv=10))
train_auPR_score_ext = np.mean(cross_val_score(ExT, train_data, train_target,scoring='average_precision',cv=10))


ExT.fit(train_data, train_target)
test_ext = ExT.predict(test_data)
test_acc_score_ext =accuracy_score(test_target, test_ext)
test_recall_score_ext = recall_score(test_target, test_ext) #sensitivtiy
test_specificity_ext = (2*test_acc_score_ext) - test_recall_score_ext #specificity or confusion matrix
test_auroc_score_ext = roc_auc_score(test_target, test_ext)
test_auPR_score_ext = average_precision_score(test_target, test_ext)
test_mcc_score_ext = matthews_corrcoef(test_target, test_ext)

print("ExtraTrees,{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},".format(train_acc_score_ext*100, train_recall_score_ext, train_specificity_score_ext, train_auroc_score_ext, train_auPR_score_ext))
print("{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},{5:3.6}\n".format(test_acc_score_ext*100, np.mean(test_recall_score_ext), np.mean(test_specificity_ext), test_auroc_score_ext, test_auPR_score_ext,test_mcc_score_ext))


#SVM linear
sv=svm.SVC(kernel='linear', C=1)
train_acc_score_sv = np.mean(cross_val_score(sv, train_data, train_target,scoring='accuracy',cv=10))
train_recall_score_sv = np.mean(cross_val_score(sv, train_data, train_target,scoring='recall',cv=10))
train_specificity_score_sv = (2*train_acc_score_sv) - train_recall_score_sv
train_auroc_score_sv = np.mean(cross_val_score(sv, train_data, train_target,scoring='roc_auc',cv=10))
train_auPR_score_sv = np.mean(cross_val_score(sv, train_data, train_target,scoring='average_precision',cv=10))


sv.fit(train_data, train_target)
test_sv = sv.predict(test_data)
test_acc_score_sv =accuracy_score(test_target, test_sv)
test_recall_score_sv = recall_score(test_target, test_sv) #sensitivtiy
test_specificity_sv = (2*test_acc_score_sv) - test_recall_score_sv #specificity or confusion matrix
test_auroc_score_sv = roc_auc_score(test_target, test_sv)
test_auPR_score_sv = average_precision_score(test_target, test_sv)
test_mcc_score_sv = matthews_corrcoef(test_target, test_sv)

print("SVM linear,{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},".format(train_acc_score_sv*100, train_recall_score_sv, train_specificity_score_sv, train_auroc_score_sv, train_auPR_score_sv))
print("{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},{5:3.6}\n".format(test_acc_score_sv*100, np.mean(test_recall_score_sv), np.mean(test_specificity_sv), test_auroc_score_sv, test_auPR_score_sv,test_mcc_score_sv))


#Logistic Regression
LR = LogisticRegression(C=1)
train_acc_score_lr = np.mean(cross_val_score(LR, train_data, train_target,scoring='accuracy',cv=10))
train_recall_score_lr = np.mean(cross_val_score(LR, train_data, train_target,scoring='recall',cv=10))
train_specificity_score_lr = (2*train_acc_score_lr) - train_recall_score_lr
train_auroc_score_lr = np.mean(cross_val_score(LR, train_data, train_target,scoring='roc_auc',cv=10))
train_auPR_score_lr = np.mean(cross_val_score(LR, train_data, train_target,scoring='average_precision',cv=10))

LR.fit(train_data, train_target)
test_lr = LR.predict(test_data)
test_acc_score_lr =accuracy_score(test_target, test_lr)
test_recall_score_lr = recall_score(test_target, test_lr) #sensitivtiy
test_specificity_lr = (2*test_acc_score_lr) - test_recall_score_lr #specificity or confusion matrix
test_auroc_score_lr = roc_auc_score(test_target, test_lr)
test_auPR_score_lr = average_precision_score(test_target, test_lr)
test_mcc_score_lr = matthews_corrcoef(test_target, test_lr)

print("Logistic Regression,{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},".format(train_acc_score_lr*100, train_recall_score_lr, train_specificity_score_lr, train_auroc_score_lr, train_auPR_score_lr))
print("{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},{5:3.6}\n".format(test_acc_score_lr*100, np.mean(test_recall_score_lr), np.mean(test_specificity_lr), test_auroc_score_lr, test_auPR_score_lr, test_mcc_score_lr))


#AdaBoost
ADAB = AdaBoostClassifier(random_state=0)
train_acc_score_adab = np.mean(cross_val_score(ADAB, train_data, train_target,scoring='accuracy',cv=10))
train_recall_score_adab = np.mean(cross_val_score(ADAB, train_data, train_target,scoring='recall',cv=10))
train_specificity_score_adab = (2*train_acc_score_adab) - train_recall_score_adab
train_auroc_score_adab = np.mean(cross_val_score(ADAB, train_data, train_target,scoring='roc_auc',cv=10))
train_auPR_score_adab = np.mean(cross_val_score(ADAB, train_data, train_target,scoring='average_precision',cv=10))

ADAB.fit(train_data, train_target)
test_adab = ADAB.predict(test_data)
test_acc_score_adab =accuracy_score(test_target, test_adab)
test_recall_score_adab = recall_score(test_target, test_adab) #sensitivtiy
test_specificity_adab = (2*test_acc_score_adab) - test_recall_score_adab #specificity or confusion matrix
test_auroc_score_adab = roc_auc_score(test_target, test_adab)
test_auPR_score_adab = average_precision_score(test_target, test_adab)
test_mcc_score_adab = matthews_corrcoef(test_target, test_adab)

print("AdaBoost,{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},".format(train_acc_score_adab*100, train_recall_score_adab, train_specificity_score_adab, train_auroc_score_adab, train_auPR_score_adab))
print("{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},{5:3.6}\n".format(test_acc_score_adab*100, np.mean(test_recall_score_adab), np.mean(test_specificity_adab), test_auroc_score_adab, test_auPR_score_adab,test_mcc_score_adab))

#DecisionTreeClassifier()
DT = DecisionTreeClassifier()
train_acc_score_dt = np.mean(cross_val_score(DT, train_data, train_target,scoring='accuracy',cv=10))
train_recall_score_dt = np.mean(cross_val_score(DT, train_data, train_target,scoring='recall',cv=10))
train_specificity_score_dt = (2*train_acc_score_dt) - train_recall_score_dt
train_auroc_score_dt = np.mean(cross_val_score(DT, train_data, train_target,scoring='roc_auc',cv=10))
train_auPR_score_dt = np.mean(cross_val_score(DT, train_data, train_target,scoring='average_precision',cv=10))


DT.fit(train_data, train_target)
test_dt = DT.predict(test_data)
test_acc_score_dt =accuracy_score(test_target, test_dt)
test_recall_score_dt = recall_score(test_target, test_dt) #sensitivtiy
test_specificity_dt = (2*test_acc_score_dt) - test_recall_score_dt#specificity or confusion matrix
test_auroc_score_dt = roc_auc_score(test_target, test_dt)
test_auPR_score_dt = average_precision_score(test_target, test_dt)
test_mcc_score_dt = matthews_corrcoef(test_target, test_dt)

print("DecisionTreeClassifier,{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},".format(train_acc_score_dt*100, train_recall_score_dt, train_specificity_score_dt, train_auroc_score_dt, train_auPR_score_dt))
print("{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},{5:3.6}\n".format(test_acc_score_dt*100, np.mean(test_recall_score_dt), np.mean(test_specificity_dt), test_auroc_score_dt, test_auPR_score_dt,test_mcc_score_dt))

#GaussianNB()
NB = GaussianNB()
train_acc_score_nb = np.mean(cross_val_score(NB, train_data, train_target,scoring='accuracy',cv=10))
train_recall_score_nb = np.mean(cross_val_score(NB, train_data, train_target,scoring='recall',cv=10))
train_specificity_score_nb = (2*train_acc_score_nb) - train_recall_score_nb
train_auroc_score_nb = np.mean(cross_val_score(NB, train_data, train_target,scoring='roc_auc',cv=10))
train_auPR_score_nb = np.mean(cross_val_score(NB, train_data, train_target,scoring='average_precision',cv=10))

NB.fit(train_data, train_target)
test_nb = NB.predict(test_data)
test_acc_score_nb =accuracy_score(test_target, test_nb)
test_recall_score_nb = recall_score(test_target, test_nb) #sensitivtiy
test_specificity_nb = (2*test_acc_score_nb) - test_recall_score_nb #specificity or confusion matrix
test_auroc_score_nb = roc_auc_score(test_target, test_nb)
test_auPR_score_nb = average_precision_score(test_target, test_nb)
test_mcc_score_nb = matthews_corrcoef(test_target, test_nb)

print("GaussianNB,{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},".format(train_acc_score_nb*100, train_recall_score_nb, train_specificity_score_nb, train_auroc_score_nb, train_auPR_score_nb))
print("{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},{5:3.6}\n".format(test_acc_score_nb*100, np.mean(test_recall_score_nb), np.mean(test_specificity_nb), test_auroc_score_nb, test_auPR_score_nb,test_mcc_score_nb))

#KNeighborsClassifier()
KNN = KNeighborsClassifier()
train_acc_score_knn = np.mean(cross_val_score(KNN, train_data, train_target,scoring='accuracy',cv=10))
train_recall_score_knn = np.mean(cross_val_score(KNN, train_data, train_target,scoring='recall',cv=10))
train_specificity_score_knn = (2*train_acc_score_knn) - train_recall_score_knn
train_auroc_score_knn = np.mean(cross_val_score(KNN, train_data, train_target,scoring='roc_auc',cv=10))
train_auPR_score_knn = np.mean(cross_val_score(KNN, train_data, train_target,scoring='average_precision',cv=10))

KNN.fit(train_data, train_target)
test_knn = KNN.predict(test_data)
test_acc_score_knn =accuracy_score(test_target, test_knn)
test_recall_score_knn = recall_score(test_target, test_knn) #sensitivtiy
test_specificity_knn = (2*test_acc_score_knn) - test_recall_score_knn #specificity or confusion matrix
test_auroc_score_knn = roc_auc_score(test_target, test_knn)
test_auPR_score_knn = average_precision_score(test_target, test_knn)
test_mcc_score_knn = matthews_corrcoef(test_target, test_knn)

print("KNeighborsClassifier,{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},".format(train_acc_score_knn*100, train_recall_score_knn, train_specificity_score_knn, train_auroc_score_knn, train_auPR_score_knn))
print("{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},{5:3.6}\n".format(test_acc_score_knn*100, np.mean(test_recall_score_knn), np.mean(test_specificity_knn), test_auroc_score_knn, test_auPR_score_knn,test_mcc_score_knn))

#LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis(n_components=500)
train_acc_score_lda = np.mean(cross_val_score(LDA, train_data, train_target,scoring='accuracy',cv=10))
train_recall_score_lda = np.mean(cross_val_score(LDA, train_data, train_target,scoring='recall',cv=10))
train_specificity_score_lda = (2*train_acc_score_lda) - train_recall_score_lda
train_auroc_score_lda = np.mean(cross_val_score(LDA, train_data, train_target,scoring='roc_auc',cv=10))
train_auPR_score_lda = np.mean(cross_val_score(LDA, train_data, train_target,scoring='average_precision',cv=10))

LDA.fit(train_data, train_target)
test_lda = LDA.predict(test_data)
test_acc_score_lda =accuracy_score(test_target, test_lda)
test_recall_score_lda = recall_score(test_target, test_lda) #sensitivtiy
test_specificity_lda = (2*test_acc_score_lda) - test_recall_score_lda #specificity or confusion matrix
test_auroc_score_lda = roc_auc_score(test_target, test_lda)
test_auPR_score_lda = average_precision_score(test_target, test_lda)
test_mcc_score_lda = matthews_corrcoef(test_target, test_lda)

print("LinearDiscriminantAnalysis,{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},".format(train_acc_score_lda*100, train_recall_score_lda, train_specificity_score_lda, train_auroc_score_lda, train_auPR_score_lda))
print("{0:3.6},{1:3.6},{2:3.6},{3:3.6},{4:3.6},{5:3.6}\n".format(test_acc_score_lda*100, np.mean(test_recall_score_lda), np.mean(test_specificity_lda), test_auroc_score_lda, test_auPR_score_lda,test_mcc_score_lda))



