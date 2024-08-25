# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score
from sklearn.svm import SVC
import pickle
from pandas import DataFrame
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sktime.transformations.panel.rocket import Rocket
from collections import Counter
from matplotlib import pyplot
from numpy import where
import statistics
from sklearn.model_selection import StratifiedKFold
from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate
import matplotlib.pyplot as plt
from datetime import datetime
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import CanonicalIntervalForest
import time

"""# 2. Data loading

### 2.1 Original Data
"""

p1_data = pd.read_pickle(r'../partition1_data.pkl')
p2_data = pd.read_pickle(r'../partition2_data.pkl')
p3_data = pd.read_pickle(r'../partition3_data.pkl')
p4_data = pd.read_pickle(r'../partition4_data.pkl')
p5_data = pd.read_pickle(r'../partition5_data.pkl')
p1_labels = pd.read_pickle(r'../partition1_labels.pkl')
p2_labels = pd.read_pickle(r'../partition2_labels.pkl')
p3_labels = pd.read_pickle(r'../partition3_labels.pkl')
p4_labels = pd.read_pickle(r'../partition4_labels.pkl')
p5_labels = pd.read_pickle(r'../partition5_labels.pkl')

# Removing B and C Class flares
result1 = np.where(p1_labels == 'B')
result2 = np.where(p2_labels == 'B')
result3 = np.where(p3_labels == 'B')
result4 = np.where(p4_labels == 'B')
result5 = np.where(p5_labels == 'B')

p1_labels = np.delete(p1_labels, result1[0], 0)
p2_labels = np.delete(p2_labels, result2[0], 0)
p3_labels = np.delete(p3_labels, result3[0], 0)
p4_labels = np.delete(p4_labels, result4[0], 0)
p5_labels = np.delete(p5_labels, result5[0], 0)

p1_data = np.delete(p1_data, result1[0], 0)
p2_data = np.delete(p2_data, result2[0], 0)
p3_data = np.delete(p3_data, result3[0], 0)
p4_data = np.delete(p4_data, result4[0], 0)
p5_data = np.delete(p5_data, result5[0], 0)

result11 = np.where(p1_labels == 'C')
result22 = np.where(p2_labels == 'C')
result33 = np.where(p3_labels == 'C')
result44 = np.where(p4_labels == 'C')
result55 = np.where(p5_labels == 'C')

p1_labels = np.delete(p1_labels, result11[0], 0)
p2_labels = np.delete(p2_labels, result22[0], 0)
p3_labels = np.delete(p3_labels, result33[0], 0)
p4_labels = np.delete(p4_labels, result44[0], 0)
p5_labels = np.delete(p5_labels, result55[0], 0)

p1_data = np.delete(p1_data, result11[0], 0)
p2_data = np.delete(p2_data, result22[0], 0)
p3_data = np.delete(p3_data, result33[0], 0)
p4_data = np.delete(p4_data, result44[0], 0)
p5_data = np.delete(p5_data, result55[0], 0)

"""### 2.2 Converting labels into Binary"""

# converting labels into binary class
p1_labels_binary = np.where(p1_labels == 'F', 'NF', p1_labels)
p1_labels_binary = np.where(p1_labels_binary == 'M', 'F', p1_labels_binary)
p1_labels_binary = np.where(p1_labels_binary == 'X', 'F', p1_labels_binary)

p2_labels_binary = np.where(p2_labels == 'F', 'NF', p2_labels)
p2_labels_binary = np.where(p2_labels_binary == 'M', 'F', p2_labels_binary)
p2_labels_binary = np.where(p2_labels_binary == 'X', 'F', p2_labels_binary)

p3_labels_binary = np.where(p3_labels == 'F', 'NF', p3_labels)
p3_labels_binary = np.where(p3_labels_binary == 'M', 'F', p3_labels_binary)
p3_labels_binary = np.where(p3_labels_binary == 'X', 'F', p3_labels_binary)

p4_labels_binary = np.where(p4_labels == 'F', 'NF', p4_labels)
p4_labels_binary = np.where(p4_labels_binary == 'M', 'F', p4_labels_binary)
p4_labels_binary = np.where(p4_labels_binary == 'X', 'F', p4_labels_binary)

p5_labels_binary = np.where(p5_labels == 'F', 'NF', p5_labels)
p5_labels_binary = np.where(p5_labels_binary == 'M', 'F', p5_labels_binary)
p5_labels_binary = np.where(p5_labels_binary == 'X', 'F', p5_labels_binary)

"""# 3. All possible partition pair"""

pp_5= [
[p1_data, p2_data, p1_labels_binary, p2_labels_binary],
[p2_data, p3_data, p2_labels_binary, p3_labels_binary],
[p3_data, p4_data, p3_labels_binary, p4_labels_binary],
[p4_data, p5_data, p4_labels_binary, p5_labels_binary],
[p5_data, p1_data, p5_labels_binary, p1_labels_binary],
]

"""# 4. Experiment

### Binary
"""

st = time.time()

cm = []

for i in pp_5:

    # Fit CLF
    clf = CanonicalIntervalForest(random_state=21)
    clf.fit(i[0], i[2])

    # Predict
    y_pred = clf.predict(i[1])

    #confusion matrix
    score = confusion_matrix(i[3],y_pred)
    cm.append(score)

tss = []


for i in cm:

    TN, FP, FN, TP = i.ravel()
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

    tss.append(TSS)


et = time.time()
ft = et-st
print("Execution Time:", ft, "seconds")

print("TSS score for CLF without B and C class labels and BINARY: \n")
print(tss)
