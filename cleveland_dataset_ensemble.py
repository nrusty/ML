#!/usr/bin/env python

## Machine Learning Final Project
## Author: Natasha Rusty                  Date: 11/29/2018
##
## Dataset: Heart Disease. Available at: https://archive.ics.uci.edu/ml/datasets/Heart+Disease [1]
## ML models: CART (decision tree), Neural Network, Deep Neural network
## Ensemble models: Bagging, Random Forests, Weighted Average
## This project aims to predict the presence of heart disease in a patient and its level
## given 14 features as described in [1].
import numpy as np
import pandas as pd
import graphviz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import random
import pickle
import cv2
import os

from graphviz import Source
from imutils import paths
from sklearn import preprocessing, tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer


from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from sklearn.ensemble import VotingClassifier

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier


# Load data
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv('data/uci-heart-disease/processed_cleveland_data_2.csv', sep=',')

# Process data
data = data.fillna(data.mean())
for item in data:                                       #converts everything to floats
    data[item] = pd.to_numeric(data[item])
x = data.values[:, 0:13]
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)                     # normalize
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
y = data.values[:, -1]
y = np.array(list(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
# Oversampling
x_train, y_train = BorderlineSMOTE(sampling_strategy='auto', kind='borderline-1').fit_resample(x_train, y_train)


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())
# Encode class values as integers
lb = LabelBinarizer()
y_train2 = lb.fit_transform(y_train)
y_test2 = lb.transform(y_test)
# Create DL model
model = Sequential()
model.add(Dense(20, input_dim=13, activation='relu'))
model.add(Dense(38, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation="softmax"))
# DNN model
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
# CART model
c = tree.DecisionTreeClassifier(criterion = "gini", random_state = 42, splitter="best",
                                max_depth=3, min_samples_leaf=2)
# ANN model
ann = MLPClassifier(solver='adam', alpha=1e-2, max_iter=300,
                    hidden_layer_sizes=(8), random_state=42)
# Ensemble: Random Forests
rf = RandomForestClassifier(bootstrap=True,  criterion='gini',
            max_depth=4,  max_leaf_nodes=None, max_features="auto",
            min_samples_split=2, n_estimators='warn', random_state=42,
            verbose=0, warm_start=False)
# Ensemble: Bagging
bagging = BaggingClassifier(rf, n_estimators=500, max_samples=1.0, random_state=42)
# Ensemble: Weighted Voting - not included in the report
eclf3 = VotingClassifier(estimators=[
       ('CART', c), ('ANN', ann), ('BAG', bagging)],
       voting='soft', weights=[14,3,3],
       flatten_transform=True)

# Training models
H = model.fit(x_train, y_train2, validation_data=(x_test, y_test2),
	epochs=120, batch_size=100)
c.fit(x_train, y_train)
ann.fit(x_train, y_train)
eclf3 = eclf3.fit(x_train, y_train)
bagging.fit(x_train, y_train)
rf.fit(x_train, y_train)

# Plot CART rules
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'] 
target_names = ['0', '1', '2', '3', '4']
dot_data = tree.export_graphviz(c, out_file='tree.dot', 
                                feature_names=feature_names,  
                                class_names=target_names,  
                                filled=True, rounded=True,  
                                special_characters=True)
Source.from_file('tree.dot')
graph = graphviz.Source(dot_data)

# Predict and print reports
predictions = model.predict(x_test, batch_size=100)
target_names = ['0', '1', '2', '3', '4']

report_dnn = classification_report(y_test2.argmax(axis=1),
	predictions.argmax(axis=1), target_names=target_names)
print(report_dnn)
print ("DeepLearning\n",confusion_matrix(y_test, predictions.argmax(axis=1)))

report_c = classification_report(y_test,	c.predict(x_test))
print(report_c)
print ("CART\n",confusion_matrix(y_test, c.predict(x_test)))

report_ann = classification_report(y_test,	ann.predict(x_test))
print(report_ann)
print ("ANN\n",confusion_matrix(y_test, ann.predict(x_test)))

print(classification_report(y_test,	eclf3.predict(x_test)))
print ("EN:3CLF\n",confusion_matrix(y_test, eclf3.predict(x_test)))

report_rf = classification_report(y_test,	rf.predict(x_test))
print(report_rf)
print ("EN:RF\n",confusion_matrix(y_test, rf.predict(x_test)))

# Ensemble:  Weighted Average - included in the report
med = [(.7*predictions.argmax(axis=1)[i] + .15*ann.predict(x_test)[i] + 
       .15*c.predict(x_test)[i]) for i in range(len(y_test))]
med = np.round(med)
report_av = classification_report(y_test,med)
print(report_av)
print ("AVERAGE\n",confusion_matrix(y_test, med))

scores = cross_val_score(bagging, x, y, cv=5)
report_bg = classification_report(y_test,	bagging.predict(x_test))
print(report_bg)
print ("Bagging\n",confusion_matrix(y_test, bagging.predict(x_test)))
print("Accuracy Bagging CART: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

# Plot the training loss and accuracy dor DNN
N = np.arange(0, 120)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('grafico') #args["plot"]

f = open("report_c.txt", "a")
f.write(report_c)

f = open("report_ann.txt", "a")
f.write(report_ann)

f = open("report_dnn.txt", "a")
f.write(report_dnn)

f = open("report_rf.txt", "a")
f.write(report_rf)

f = open("report_av.txt", "a")
f.write(report_av)

f = open("report_bg.txt", "a")
f.write(report_bg)
