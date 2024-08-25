# -*- coding: utf-8 -*-

# PREREQUISITES


# IMPORTS

import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score
import pickle
from pandas import DataFrame
from collections import Counter
from matplotlib import pyplot
from numpy import where
import statistics
import matplotlib.pyplot as plt
from datetime import datetime
!pip install tensorflow
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import optimizers, Sequential, Model
from tensorflow.keras.layers import LSTM
import tensorflow.keras
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# DATA LOADING

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

# CONVERTING DATA TO BINARY NUMERICALS

p1_labels_binary = np.where(p1_labels == 'B', 0, p1_labels)
p1_labels_binary = np.where(p1_labels_binary == 'C', 0, p1_labels_binary)
p1_labels_binary = np.where(p1_labels_binary == 'F', 0, p1_labels_binary)
p1_labels_binary = np.where(p1_labels_binary == 'M', 1, p1_labels_binary)
p1_labels_binary = np.where(p1_labels_binary == 'X', 1, p1_labels_binary)

p2_labels_binary = np.where(p2_labels == 'B', 0, p2_labels)
p2_labels_binary = np.where(p2_labels_binary == 'C', 0, p2_labels_binary)
p2_labels_binary = np.where(p2_labels_binary == 'F', 0, p2_labels_binary)
p2_labels_binary = np.where(p2_labels_binary == 'M', 1, p2_labels_binary)
p2_labels_binary = np.where(p2_labels_binary == 'X', 1, p2_labels_binary)

p3_labels_binary = np.where(p3_labels == 'B', 0, p3_labels)
p3_labels_binary = np.where(p3_labels_binary == 'C', 0, p3_labels_binary)
p3_labels_binary = np.where(p3_labels_binary == 'F', 0, p3_labels_binary)
p3_labels_binary = np.where(p3_labels_binary == 'M', 1, p3_labels_binary)
p3_labels_binary = np.where(p3_labels_binary == 'X', 1, p3_labels_binary)

p4_labels_binary = np.where(p4_labels == 'B', 0, p4_labels)
p4_labels_binary = np.where(p4_labels_binary == 'C', 0, p4_labels_binary)
p4_labels_binary = np.where(p4_labels_binary == 'F', 0, p4_labels_binary)
p4_labels_binary = np.where(p4_labels_binary == 'M', 1, p4_labels_binary)
p4_labels_binary = np.where(p4_labels_binary == 'X', 1, p4_labels_binary)

p5_labels_binary = np.where(p5_labels == 'B', 0, p5_labels)
p5_labels_binary = np.where(p5_labels_binary == 'C', 0, p5_labels_binary)
p5_labels_binary = np.where(p5_labels_binary == 'F', 0, p5_labels_binary)
p5_labels_binary = np.where(p5_labels_binary == 'M', 1, p5_labels_binary)
p5_labels_binary = np.where(p5_labels_binary == 'X', 1, p5_labels_binary)

"""# EXPERIMENTS

# Pair 1
"""

X_train, y_train, X_test, y_test = p1_data, p1_labels_binary, p2_data, p2_labels_binary
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_train = X_train / np.linalg.norm(X_train)
X_test = X_test / np.linalg.norm(X_test)

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(24,60)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, y_train, epochs=5)

pred = model.predict(X_test)

pp = []
for j in pred:
    if (j < 0.5):
        pp.append(0)
    else:
        pp.append(1)

score = confusion_matrix(y_test,pp)

print(score)

TN, FP, FN, TP = score.ravel()
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

print(TSS)

"""# Pair 2"""

X_train, y_train, X_test, y_test = p2_data, p2_labels_binary, p3_data, p3_labels_binary
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_train = X_train / np.linalg.norm(X_train)
X_test = X_test / np.linalg.norm(X_test)

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(24,60)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, y_train, epochs=5)

pred = model.predict(X_test)

pp = []
for j in pred:
    if (j < 0.5):
        pp.append(0)
    else:
        pp.append(1)

score = confusion_matrix(y_test,pp)

print(score)

TN, FP, FN, TP = score.ravel()
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

print(TSS)

"""# Pair 3"""

X_train, y_train, X_test, y_test = p3_data, p3_labels_binary, p4_data, p4_labels_binary
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_train = X_train / np.linalg.norm(X_train)
X_test = X_test / np.linalg.norm(X_test)

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(24,60)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, y_train, epochs=5)

pred = model.predict(X_test)

pp = []
for j in pred:
    if (j < 0.5):
        pp.append(0)
    else:
        pp.append(1)

score = confusion_matrix(y_test,pp)

print(score)

TN, FP, FN, TP = score.ravel()
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

print(TSS)

"""# Pair 4

"""

X_train, y_train, X_test, y_test = p4_data, p4_labels_binary, p5_data, p5_labels_binary
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_train = X_train / np.linalg.norm(X_train)
X_test = X_test / np.linalg.norm(X_test)

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(24,60)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, y_train, epochs=5)

pred = model.predict(X_test)

pp = []
for j in pred:
    if (j < 0.5):
        pp.append(0)
    else:
        pp.append(1)

score = confusion_matrix(y_test,pp)

print(score)

TN, FP, FN, TP = score.ravel()
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

print(TSS)

"""# Pair 5


"""

X_train, y_train, X_test, y_test = p5_data, p5_labels_binary, p1_data, p1_labels_binary
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_train = X_train / np.linalg.norm(X_train)
X_test = X_test / np.linalg.norm(X_test)

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(24,60)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, y_train, epochs=5)

pred = model.predict(X_test)

pp = []
for j in pred:
    if (j < 0.5):
        pp.append(0)
    else:
        pp.append(1)

score = confusion_matrix(y_test,pp)

print(score)

TN, FP, FN, TP = score.ravel()
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

print(TSS)
