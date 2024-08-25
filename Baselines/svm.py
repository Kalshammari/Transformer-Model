# -*- coding: utf-8 -*-


!pip install scipy
import json
import pandas as pd
import numpy as np
from numpy import array
from numpy import hstack
import scipy.stats as st
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import numpy as np
!pip install sktime
from sktime.datasets import load_airline
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas import DataFrame
from datetime import datetime
from sklearn.linear_model import RidgeClassifierCV
!pip install tsai
from tsai.models.MINIROCKET import *
from tsai.basics import *
import scipy.stats as stats
from sklearn.model_selection import TimeSeriesSplit
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_basic_motions
from sktime.transformations.panel.compose import ColumnConcatenator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sktime.transformations.panel.rocket import Rocket
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot
from numpy import where


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
import statistics
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
!pip install --upgrade pip

"""# Data loading"""

p1_data = pd.read_csv('p1_data_new.csv', index_col=0)
p2_data = pd.read_csv('p2_data_new.csv', index_col=0)
p3_data = pd.read_csv('p3_data_new.csv', index_col=0)
p4_data = pd.read_csv('p4_data_new.csv', index_col=0)
p5_data = pd.read_csv('p5_data_new.csv', index_col=0)
p1_labels = pd.read_pickle(r'partition1_labels.pkl')
p2_labels = pd.read_pickle(r'partition2_labels.pkl')
p3_labels = pd.read_pickle(r'partition3_labels.pkl')
p4_labels = pd.read_pickle(r'partition4_labels.pkl')
p5_labels = pd.read_pickle(r'partition5_labels.pkl')

"""# Removing B and C class flares"""

print("Partition 1 Data and Labels Shape: ")
print(p1_data.shape)
print(p1_labels.shape)
print(np.unique(p1_labels, return_counts=1))
print("\n")


print("Partition 2 Data and Labels Shape: ")
print(p2_data.shape)
print(p2_labels.shape)
print(np.unique(p2_labels, return_counts=1))
print("\n")

print("Partition 3 Data and Labels Shape: ")
print(p3_data.shape)
print(p3_labels.shape)
print(np.unique(p3_labels, return_counts=1))
print("\n")

print("Partition 4 Data and Labels Shape: ")
print(p4_data.shape)
print(p4_labels.shape)
print(np.unique(p4_labels, return_counts=1))
print("\n")

print("Partition 5 Data and Labels Shape: ")
print(p5_data.shape)
print(p5_labels.shape)
print(np.unique(p5_labels, return_counts=1))

# Removing B and C class

# Adding Target to main data
p1_data['Target'] = p1_labels
p2_data['Target'] = p2_labels
p3_data['Target'] = p3_labels
p4_data['Target'] = p4_labels
p5_data['Target'] = p5_labels

# Filtering F class Target from main data
p1_data = p1_data[p1_data['Target'] != 'B']
p2_data = p2_data[p2_data['Target'] != 'B']
p3_data = p3_data[p3_data['Target'] != 'B']
p4_data = p4_data[p4_data['Target'] != 'B']
p5_data = p5_data[p5_data['Target'] != 'B']
p1_data = p1_data[p1_data['Target'] != 'C']
p2_data = p2_data[p2_data['Target'] != 'C']
p3_data = p3_data[p3_data['Target'] != 'C']
p4_data = p4_data[p4_data['Target'] != 'C']
p5_data = p5_data[p5_data['Target'] != 'C']

# Splitting labels back from data
p1_labels = p1_data['Target'].values
p1_data = p1_data.drop(['Target'],axis=1).values

p2_labels = p2_data['Target'].values
p2_data = p2_data.drop(['Target'],axis=1).values

p3_labels = p3_data['Target'].values
p3_data = p3_data.drop(['Target'],axis=1).values

p4_labels = p4_data['Target'].values
p4_data = p4_data.drop(['Target'],axis=1).values

p5_labels = p5_data['Target'].values
p5_data = p5_data.drop(['Target'],axis=1).values

"""After removing F class"""

print("Partition 1 Data and Labels Shape: ")
print(p1_data.shape)
print(p1_labels.shape)
print(np.unique(p1_labels, return_counts=1))
print("\n")


print("Partition 2 Data and Labels Shape: ")
print(p2_data.shape)
print(p2_labels.shape)
print(np.unique(p2_labels, return_counts=1))
print("\n")

print("Partition 3 Data and Labels Shape: ")
print(p3_data.shape)
print(p3_labels.shape)
print(np.unique(p3_labels, return_counts=1))
print("\n")

print("Partition 4 Data and Labels Shape: ")
print(p4_data.shape)
print(p4_labels.shape)
print(np.unique(p4_labels, return_counts=1))
print("\n")

print("Partition 5 Data and Labels Shape: ")
print(p5_data.shape)
print(p5_labels.shape)
print(np.unique(p5_labels, return_counts=1))

"""# Converting labels into binary"""

# Partition 1 Labels
p1_labels_binary = np.where(p1_labels == 'F', 'NF', p1_labels)
p1_labels_binary = np.where(p1_labels_binary == 'M', 'F', p1_labels_binary)
p1_labels_binary = np.where(p1_labels_binary == 'X', 'F', p1_labels_binary)

# Partition 2 Labels
p2_labels_binary = np.where(p2_labels == 'F', 'NF', p2_labels)
p2_labels_binary = np.where(p2_labels_binary == 'M', 'F', p2_labels_binary)
p2_labels_binary = np.where(p2_labels_binary == 'X', 'F', p2_labels_binary)

# Partition 3 Labels
p3_labels_binary = np.where(p3_labels == 'F', 'NF', p3_labels)
p3_labels_binary = np.where(p3_labels_binary == 'M', 'F', p3_labels_binary)
p3_labels_binary = np.where(p3_labels_binary == 'X', 'F', p3_labels_binary)

# Partition 4 Labels
p4_labels_binary = np.where(p4_labels == 'F', 'NF', p4_labels)
p4_labels_binary = np.where(p4_labels_binary == 'M', 'F', p4_labels_binary)
p4_labels_binary = np.where(p4_labels_binary == 'X', 'F', p4_labels_binary)

# Partition 5 Labels
p5_labels_binary = np.where(p5_labels == 'F', 'NF', p5_labels)
p5_labels_binary = np.where(p5_labels_binary == 'M', 'F', p5_labels_binary)
p5_labels_binary = np.where(p5_labels_binary == 'X', 'F', p5_labels_binary)

"""# All possible partition pair

Binary Labels
"""

pp_5 = [
[p1_data, p2_data, p1_labels_binary, p2_labels_binary],
[p2_data, p3_data, p2_labels_binary, p3_labels_binary],
[p3_data, p4_data, p3_labels_binary, p4_labels_binary],
[p4_data, p5_data, p4_labels_binary, p5_labels_binary],
[p5_data, p1_data, p5_labels_binary, p1_labels_binary],
]

"""# Experiment 1 : Binary"""

st = time.time()
acc = []
cm = []

pair = 1

for i in pp_5:

    print("Pair Number: ", pair)
    pair+=1

    svc=SVC()

    # fit classifier to training set
    svc.fit(i[0],i[2])

    # make predictions on test set
    y_pred=svc.predict(i[1])

    # compute and print accuracy score
    score = accuracy_score(i[3], y_pred)
    acc.append(score)
    print(score)

    #confusion matrix
    score2 = confusion_matrix(i[3],y_pred)
    cm.append(score2)
    print(score2)
    print("\n")

et = time.time()
ft = et-st
print("Execution Time:", ft, "seconds")

"""Measures"""

tss = []
order = 1
for i in cm:
    TP, FN, FP, TN = i.ravel()
    TN = np.array(TN)
    FP = np.array(FP)
    FN = np.array(FN)
    TP = np.array(TP)
    acc = (TP + TN) / (TP + FN + TN + FP)
    pr_pos = TP/(TP + FP)
    pr_neg = TN/(TN + FN)
    rc_pos = TP/(TP + FN)
    rc_neg = TN/(TN + FP)
    f1_pos = (2 * pr_pos * rc_pos) / (pr_pos + rc_pos)
    f1_neg = (2 * pr_neg * rc_neg) / (pr_neg + rc_neg)
    P = TP + FN
    N = TN + FP
    HSS1 = (TP + TN - N) / P
    HSS2 = (2*((TP*TN)-(FP*FN)))/(P*(FN+TN)+(TP+FP)*N)
    CH = ((TP+FP)*(TP+FN))/(P+N)
    GS = (TP-CH)/(TP+FP+FN-CH)
    TSS = ((TP*TN)-(FP*FN))/(P*N)
    measures_array = ["Acc:", acc, "pr_pos:", pr_pos, "pr_neg:", pr_neg, "rc_pos:", rc_pos, "rc_neg:", rc_neg, "f1_pos:", f1_pos, "f1_neg:", f1_neg, "HSS1:", HSS1, "HSS2:", HSS2, "GS:", GS, "TSS:", TSS]
    tss.append(TSS)
    print("Pair Number: ",order)
    order += 1
    print(measures_array)
    print("\n")

print("TSS of binary labels: ",tss)
