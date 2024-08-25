# -*- coding: utf-8 -*-


import tensorflow as tf
tf.config.list_physical_devices('GPU')
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_name = tf.test.gpu_device_name()
print(tf.config.list_physical_devices('GPU'))
#print('Found GPU at: {}'.format(device_name))

#Import The Solar Flare Data Set Files

import numpy as np
import pandas as pd
import networkx as nx
import pickle
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import plot_model
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score

def loadInputs(file_name):
        with open(file_name, 'rb') as fp:
            obj = pickle.load(fp)
        return obj

Sampled_train_inputs=loadInputs("partition1_data_Binary_no_BC.pkl")# you can change it to different data files
Sampled_train_labels=loadInputs("partition1_labels_Binary_no_BC.pkl")
temp=Sampled_train_inputs[0]
print(temp)
trainData = Sampled_train_inputs
trainLabel = Sampled_train_labels
print("trainData.shape: ", trainData.shape)
print("trainLebel.shape: ", trainLabel.shape)
print("Classes/labels : ",np.unique(trainLabel))

Sampled_test_inputs=loadInputs("partition2_data_Binary_no_BC.pkl")# you can change it to different data files
Sampled_test_labels=loadInputs("partition2_labels_Binary_no_BC.pkl")
temp=Sampled_test_inputs[0]
print(temp)
testData = Sampled_test_inputs
testLabel = Sampled_test_labels
print("trainData.shape: ", testData.shape)
print("trainLebel.shape: ", testLabel.shape)
print("Classes/labels : ",np.unique(testLabel))

#standardization/z normalization of the univaraite time series
#-------------------data transform 3D->2D->3D ------------------------------
#Takes 3D array(x,y,z) >> transpose(y,z) >> return (x,z,y)
def GetTransposed2D(arrayFrom):
    toReturn = []
    alen = arrayFrom.shape[0]
    for i in range(0, alen):
        toReturn.append(arrayFrom[i].T)

    return np.array(toReturn)

#Takes 3D array(x,y,z) >> Flatten() >> return (x*y,z)
def Make2D(array3D):
    toReturn = []
    x = array3D.shape[0]
    y = array3D.shape[1]
    for i in range(0, x):
        for j in range(0, y):
            toReturn.append(array3D[i,j])

    return np.array(toReturn)

#Transform instance(92400, 33) into(1540x60x33)
def Get3D_MVTS_from2D(array2D, windowSize):
    arrlen = array2D.shape[0]
    mvts = []
    for i in range(0, arrlen, windowSize):
        mvts.append(array2D[i:i+windowSize])

    return np.array(mvts)




#-------------------data Scaler ------------------------------------------
from sklearn.preprocessing import StandardScaler

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized

def GetStandardScaler(data2d):
    scaler = StandardScaler()
    scaler = scaler.fit(data2d)
    return scaler

def GetStandardScaledData(data2d):
    scaler = StandardScaler()
    scaler = scaler.fit(data2d)
    #print(scaler.mean_)
    data_scaled = scaler.transform(data2d)
    return data_scaled

def transform_scale_data(data3d, scaler):
    print("original data shape:", data3d.shape)
    trans = GetTransposed2D(data3d)
    print("transposed data shape:", trans.shape)    #(x, 60, 33)
    data2d = Make2D(trans)
    print("2d data shape:", data2d.shape)
    #  scaler = GetStandardScaler(data2d)
    data_scaled = scaler.transform(data2d)
    mvts_scalled = Get3D_MVTS_from2D(data_scaled, data3d.shape[2])#,60)
    print("mvts data shape:", mvts_scalled.shape)
    transBack = GetTransposed2D(mvts_scalled)
    print("transBack data shape:", transBack.shape)
    return transBack

TORCH_SEED = 0
#building standard scaler on train data X

#---------------Node Label Data Scaling-----------
trans = GetTransposed2D(trainData)
data2d = Make2D(trans)
scaler = GetStandardScaler(data2d)

trainData = transform_scale_data(trainData, scaler)
#trainLabel = trainLabel
unique_y_train, counts_y_train = np.unique(trainLabel, return_counts=True)
num_y_class = unique_y_train.shape[0]
print("X_train shape: ", trainData.shape)
print("y_train shape: ", trainLabel.shape)
#y_train_stats = dict(zip(unique_y_train, counts_y_train))
print("unique_y_train: ", unique_y_train)
print("y_train_counts: ", counts_y_train)
print("num_y_class: ", num_y_class)

X_test = transform_scale_data(testData, scaler)

#y_test = np.array(utils.get_int_labels_from_str(y_test))

y_test = testLabel

print("type(X_test): ", type(X_test), " X_test.shape: ",X_test.shape)

print("type(y_test): ", type(y_test), " y_test.shape: ",y_test.shape)
unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)

num_y_classes = unique_y_train.shape[0]

print("X_test shape: ", X_test.shape)

print("y_test shape: ", y_test.shape)

#y_train_stats = dict(zip(unique_y_train, counts_y_train))

print("unique_y_test: ", unique_y_test)

print("y_test_counts: ", counts_y_test)

#Transposing trainData to shape:(len(X_train), 60, 24)

trainDatatemp=np.empty([len(trainData),60, 24])

n=len(trainData)

for l in range(0, n):

  temp=trainData[l]

  temp=temp.T

  trainDatatemp[l,:,:]=temp





X_train=trainDatatemp

print("Transposing trainData shape: ",X_train.shape)
y_train=trainLabel

#Transposing testData to shape:(len(X_test), 60, 24)

testDatatemp=np.empty([len(X_test),60, 24])

n=len(X_test)

for l in range(0, n):

  temp=X_test[l]

  temp=temp.T

  testDatatemp[l,:,:]=temp





X_test=testDatatemp

print("Transposing testData shape: ",X_test.shape)

def TSS(c,valLabel, vpredictaedLabel):
         measures_array = np.zeros((24, 11))
         max_val_classification_report_dict=metrics.classification_report(valLabel, vpredictaedLabel, digits=6,output_dict=True)
         TN, FP, FN, TP = metrics.confusion_matrix(valLabel, vpredictaedLabel).ravel()
         acc = (TP + TN) / (TP + FN + TN + FP)
         pr_pos = TP / (TP + FP)
         pr_neg = TN / (TN + FN)
         rc_pos = TP / (TP + FN)
         rc_neg = TN / (TN + FP)
         f1_pos = (2 * pr_pos * rc_pos) / (pr_pos + rc_pos)
         f1_neg = (2 * pr_neg * rc_neg) / (pr_neg + rc_neg)
         P = TP + FN
         N = TN + FP
         HSS1 = (TP + TN - N) / P
         HSS2 = (2 * ((TP * TN) - (FP * FN))) / (P * (FN + TN) + (TP + FP) * N)
         CH = ((TP + FP) * (TP + FN)) / (P + N)
         GS = (TP - CH) / (TP + FP + FN - CH)
         TSS = ((TP * TN) - (FP * FN)) / (P * N)

         print("TP", TP,"FP", FP)
         print("FN", FN,"TN", TN)
         print("TSS", TSS)

         measures_array[c, :] = [acc, pr_pos, pr_neg, rc_pos, rc_neg, f1_pos, f1_neg, HSS1, HSS2, GS, TSS]
         print( "acc, pr_pos, pr_neg, rc_pos, rc_neg, f1_pos, f1_neg, HSS1, HSS2, GS, TSS for Parameter Number:",c,":" ,measures_array[c, :])
         return TSS,measures_array[c, :]

from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

def transformer_encoder(inputs, head_size, num_heads, ff_dim):
    # Normalization and Attention
    #x=inputs
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads
    )(x, x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
):
    n_classes=len(unique_y_train)
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        #x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

from sklearn.manifold import TSNE
import time
from datetime import timedelta
start_time = time.monotonic()

#Train and evaluate
input_shape = X_train.shape[1:]
MeasuresArrayTestTss=[]
TestTss=[]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=10,
    mlp_units=[64],

)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

for r in range(0,1):

    print("Random_state: ", r)


    print("X_train.shape y_train.shape y_test.shape ",X_train.shape, y_train.shape)
    print("X_test.shape y_test.shape ",X_test.shape, y_test.shape)#check percentage of examples
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)
    unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
    y_train_stats = dict(zip(unique_y_train, counts_y_train))
    print("y_train_counts")
    print(y_train_stats)

    unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
    y_test_stats = dict(zip(unique_y_test, counts_y_test))
    print("y_test_counts")
    print(y_test_stats)


    history= model.fit(
    X_train,
    y_train,
    validation_split=0.10,
    epochs=5,
    batch_size=8,
    callbacks=callbacks,)
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    model.evaluate(X_test, y_test, verbose=1)
    # Plot the model
    plot_model(model, to_file='transformer_model.png', show_shapes=True, show_layer_names=True)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis = 1)


    n_classes=len(unique_y_train)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=range(n_classes), yticklabels=range(n_classes))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    target_names=[0,1]
    report = classification_report(y_test, y_pred, target_names=target_names,output_dict=True)
    # Extract metrics
    precision = [report[label]['precision'] for label in target_names]
    recall = [report[label]['recall'] for label in target_names]
    f1_score = [report[label]['f1-score'] for label in target_names]
    # Plot the metrics
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(target_names))
    width = 0.3
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1_score, width, label='F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(target_names)
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Classification Report Metrics')
    ax.legend()
    plt.tight_layout()
    plt.show()
    testTSS,MATSS=TSS(r,y_test, y_pred)
    MeasuresArrayTestTss.append(MATSS)
    TestTss.append(testTSS)
    end_time = time.monotonic()
    print("Time takes to run the code:",timedelta(seconds=end_time - start_time))

file1 = open("P2_MeasuresArraytest.txt", "w")
df1=pd.DataFrame(MeasuresArrayTestTss)
file1.write("acc, pr_pos, pr_neg, rc_pos, rc_neg, f1_pos, f1_neg, HSS1, HSS2, GS, TSS")
file1.write(df1.to_string())
file1.write("---------------------------------------")
file1.close()
