import numpy as np
from matplotlib import pyplot as plt
#lets perform stochastic gradient descent to learn the seperating hyperplane between both classes

def svm_sgd(X, Y):
    #Initialize our SVMs weight vector with zeros (3 values)
    w = np.zeros(len(X[0]))
    #The learning rate
    eta = 1
    #how many iterations to train for
    epochs = 100000
    #store misclassifications so we can plot how they change over time
    errors = []

    #training part, gradient descent part
    for epoch in range(1,epochs):
        error = 0
        for i, x in enumerate(X):
            #misclassification
            if (Y[i]*np.dot(X[i], w)) < 1:
                #misclassified update for ours weights
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
                error = 1
                
            else:
                #correct classification, update our weights
                w = w + eta * (-2  *(1/epoch)* w)
        errors.append(error)
    return w, errors

def svm_sgd_plot(X,Y,w,errors):
    #lets plot the rate of classification errors during training for our SVM
    plt.plot(errors, '|')
    plt.ylim(0.5,1.5)
    #plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()

    for i, sample in enumerate(X):
        if Y[i] != 1:
            plt.scatter(sample[0], sample[1], s=120, linewidths=2, color = "red")
        else:
            plt.scatter(sample[0], sample[1], s=120, linewidths=2, color =  "green")
    # Print the hyperplane calculated by svm_sgd()
    x2=[w[0],w[1],-w[1],w[0]]
    x3=[w[0],w[1],w[1],-w[0]]
    x2x3 =np.array([x2,x3])
    X,Y,U,V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X,Y,U,V,scale=1, color='blue')

    plt.show()

def plot_X_Y(X,Y):
    for i, sample in enumerate(X):
        if Y[i] != 1:
            plt.scatter(sample[0], sample[1], s=120, linewidths=2, color = "red")
        else:
            plt.scatter(sample[0], sample[1], s=120, linewidths=2, color =  "green")
    plt.show()


