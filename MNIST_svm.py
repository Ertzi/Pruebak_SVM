# MNIST classification using Support Vector algorithm with RBF kernel
# Author: Krzysztof Sopyla <krzysztofsopyla@gmail.com>
# https://ksopyla.com
# License: MIT

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
import tempfile
import sklearn
import pandas as pd

# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
#fetch original mnist dataset
from sklearn.datasets import fetch_openml

# import custom module
from mnist_helpers import *


# it creates mldata folder in your root project folder
#mnist = fetch_openml("mnist_784",version = 1, cache = True)

#minist object contains: data, COL_NAMES, DESCR, target fields
#you can check it by running
#print(mnist.keys())

#data field is 70k x 784 array, each row represents pixels from 28x28=784 image
#images = mnist.data
#targets = mnist.target

training = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

print(training.head())


images = 1
targets = 1

# Let's have a look at the random 16 images, 
# We have to reshape each data row, from flat array of 784 int to 28x28 2D array

#pick  random indexes from 0 to size of our dataset
show_some_digits(images,targets)


#---------------- classification begins -----------------
#scale data for [0,255] -> [0,1]
#sample smaller size for testing
#rand_idx = np.random.choice(images.shape[0],10000)
#X_data =images[rand_idx]/255.0
#Y      = targets[rand_idx]

#full dataset classification
X_data = images/255.0
Y = targets

#split data to train and test
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)


################ Classifier with good params ###########
# Create a classifier: a support vector classifier

param_C = 5
param_gamma = 0.05
classifier = svm.SVC(C=param_C,gamma=param_gamma)

#We learn the digits on train part
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
classifier.fit(X_train, y_train)
end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))


########################################################
# Now predict the value of the test
expected = y_test
predicted = classifier.predict(X_test)

show_some_digits(X_test,predicted,title_text="Predicted {}")

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
      
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))
