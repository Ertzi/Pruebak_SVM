import numpy as np
from SVM_with_SGD_algorithm import svm_sgd,svm_sgd_plot, plot_X_Y
import random

random.seed(100)

# Ejemplo 1:
X1 = np.array([[-2,4,-1],[4,1,-1],[1, 6, -1],[2, 4, -1],[6, 2, -1],])
Y1 = np.array([-1,-1,1,1,1])
w1,errors1 = svm_sgd(X1,Y1)
svm_sgd_plot(X1,Y1,w1,errors1)

# Ejemplo 2:
X2 = np.array([[1,2,-1],[3,4,-1],[2,6,-1],[1,-5,-1],[-2,3,-1],[5,-5,-1],[-6,1,-1],[8,3,-1],[2,2,-1],[2,-3,-1],[-4,-1,-1],[0,2,-1]])
Y2 = np.array([1,1,1,-1,1,-1,-1,1,1,-1,-1,1])
w2,errors2 = svm_sgd(X2,Y2)
svm_sgd_plot(X2,Y2,w2,errors2)

# Ejemplo 3 (Generación automática de puntos):
n = 15# Cantidad de puntos
dx = 15 # Distancia a la que movemos todos los puntos en el eje x
dy = 5
X3_1 = np.zeros((n,3))
Y3_1 = np.zeros(n)
for i in range(n):
    X3_1[i] = [random.randint(-10,10)-dx, random.randint(-10,10)-dy,-1]
    Y3_1[i] = 1
X3_2 = np.zeros((n,3))
Y3_2 = np.zeros(n)
for i in range(n):
    X3_2[i] = [random.randint(-10,10)+dx, random.randint(-10,10)+dy,-1]
    Y3_2[i] = -1

X3 = np.zeros((2*n,3))
Y3 = np.zeros(2*n)

for i in range(n):
    X3[2*i+1] = X3_1[i]
    X3[2*i] = X3_2[i]
    Y3[2*i+1] = Y3_1[i]
    Y3[2*i] = Y3_2[i]

plot_X_Y(X3,Y3)
w3,errors3 = svm_sgd(X3,Y3)
svm_sgd_plot(X3,Y3,w3,errors3)


