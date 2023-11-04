# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
import tempfile
import sklearn
import pandas as pd

# Import datasets, classifiers and performance metrics
from sklearn.svm import SVC
#fetch original mnist dataset
from sklearn.datasets import fetch_openml

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

model = SVC()
model.fit(X_train,Y_train)

print(model.score(X_test,Y_test))
