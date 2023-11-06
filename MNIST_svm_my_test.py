# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
import tempfile
import sklearn
import pandas as pd

# Import datasets, classifiers and performance metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay
#fetch original mnist dataset

# import custom module
from mnist_helpers import *


training = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")


print(training.head())
print(training.describe())

Y_train = training["label"]
X_train = training.iloc[:,1:]

Y_test = test["label"]
X_test = test.iloc[:,1:]


# TRAIN THE MODELS

#CHANGE C PARAMETER
model1 = SVC(C = 1)
#model1.fit(X_train,Y_train)
#print(f"Model C=1 score: {model1.score(X_test,Y_test)}")

model2 = SVC(C = 2)
#model2.fit(X_train,Y_train)
#print(f"Model C=2 score: {model2.score(X_test,Y_test)}")

model3 = SVC(C = 3)
#model3.fit(X_train,Y_train)
#print(f"Model C=3 score: {model3.score(X_test,Y_test)}")

model4 = SVC(C = 4) # OPTIMAL
model4.fit(X_train,Y_train)
print(f"Model C=4 score: {model4.score(X_test,Y_test)}")

model5 = SVC(C = 5)
#model5.fit(X_train,Y_train)
#print(f"Model C=5 score: {model5.score(X_test,Y_test)}")

model6 = SVC(C = 6)
#model6.fit(X_train,Y_train)
#print(f"Model C=6 score: {model6.score(X_test,Y_test)}")

model7 = SVC(C = 7)
#model7.fit(X_train,Y_train)
#print(f"Model C=7 score: {model7.score(X_test,Y_test)}")

model8 = SVC(C = 8)
#model8.fit(X_train,Y_train)
#print(f"Model C=8 score: {model8.score(X_test,Y_test)}")

model9 = SVC(C = 9)
#model9.fit(X_train,Y_train)
#print(f"Model C=9 score: {model9.score(X_test,Y_test)}")

model10 = SVC(C = 10)
#model10.fit(X_train,Y_train)
#print(f"Model C=10 score: {model10.score(X_test,Y_test)}")


# LINEAR SVM
linear_model = LinearSVC(C = 4,max_iter=1)
#linear_model.fit(X_train,Y_train)
#print(f"Linear model score: {linear_model.score(X_test,Y_test)}")

# LINEAR KERNEL SVM
linear_kernel = SVC(C = 4,kernel="linear")
#linear_kernel.fit(X_train,Y_train)
#print(f"Linear kernel score: {linear_kernel.score(X_test,Y_test)}")

# CONFUSION MATRIX:
fig, ax = plt.subplots(figsize=(10, 5))
predict4 = model4.predict(X_test)
ConfusionMatrixDisplay.from_predictions(Y_test, predict4, ax=ax)
ax.xaxis.set_ticklabels(["0","1","2","3","4","5","6","7","8","9"])
ax.yaxis.set_ticklabels(["0","1","2","3","4","5","6","7","8","9"])
plt.show()

#fig, ax = plt.subplots(figsize=(10, 5))
#linear_predict = linear_model.predict(X_test)
#ConfusionMatrixDisplay.from_predictions(Y_test, linear_predict, ax=ax)
#ax.xaxis.set_ticklabels(["0","1","2","3","4","5","6","7","8","9"])
#ax.yaxis.set_ticklabels(["0","1","2","3","4","5","6","7","8","9"])
#plt.show()

#fig, ax = plt.subplots(figsize=(10, 5))
#linear_kernel_predict = linear_kernel.predict(X_test)
#ConfusionMatrixDisplay.from_predictions(Y_test, linear_kernel_predict, ax=ax)
#ax.xaxis.set_ticklabels(["0","1","2","3","4","5","6","7","8","9"])
#ax.yaxis.set_ticklabels(["0","1","2","3","4","5","6","7","8","9"])
#plt.show()

