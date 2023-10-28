import numpy as np
import matplotlib.pyplot as plt

# Función de kernel lineal
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# Clase para representar el modelo SVM
class SoftMarginSVM:
    def __init__(self, C=1.0, max_iter=100, tol=1e-3):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol




    def fit(self, X, y):
        m, n = X.shape
        self.X = X
        self.y = y
        self.alphas = np.zeros(m)
        self.b = 0

        for iteration in range(self.max_iter):
            num_changed_alphas = 0
            for i in range(m):
                Ei = self.decision_function(X[i]) - y[i]
                if (y[i] * Ei < -self.tol and self.alphas[i] < self.C) or (y[i] * Ei > self.tol and self.alphas[i] > 0):
                    j = np.random.choice([k for k in range(m) if k != i])
                    Ej = self.decision_function(X[j]) - y[j]

                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                        H = min(self.C, self.alphas[j] + self.alphas[i])

                    if L == H:
                        continue

                    eta = 2 * linear_kernel(X[i], X[j]) - linear_kernel(X[i], X[i]) - linear_kernel(X[j], X[j])
                    if eta >= 0:
                        continue

                    self.alphas[j] -= y[j] * (Ei - Ej) / eta
                    self.alphas[j] = min(H, max(L, self.alphas[j]))

                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])

                    b1 = self.b - Ei - y[i] * (self.alphas[i] - alpha_i_old) * linear_kernel(X[i], X[i]) - y[j] * (self.alphas[j] - alpha_j_old) * linear_kernel(X[i], X[j])
                    b2 = self.b - Ej - y[i] * (self.alphas[i] - alpha_i_old) * linear_kernel(X[i], X[j]) - y[j] * (self.alphas[j] - alpha_j_old) * linear_kernel(X[j], X[j])

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                break

    def decision_function(self, x):
        return np.dot(self.alphas * self.y, [linear_kernel(x, xi) for xi in self.X]) + self.b

# EJEMPLO 1
if True:
    # Crear un conjunto de datos de ejemplo
    X = np.array([[3, 3], [4, 4], [1, 1], [2, 2]])
    y = np.array([1, 1, -1, -1])

    # Crear y entrenar el modelo SVM
    svm = SoftMarginSVM(C=1.0, max_iter=100, tol=1e-3)
    svm.fit(X, y)

    # Imprimir los multiplicadores de Lagrange y el término de sesgo
    print("Multiplicadores de Lagrange (alphas):", svm.alphas)
    print("Término de sesgo (b):", svm.b)

    # Dibujar los puntos de datos
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    # Dibujar el hiperplano de decisión
    ax = plt.gca()
    xlim = ax.get_xlim()
    w = (svm.alphas * y).dot(X)
    b = svm.b
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = (-w[0] * xx - b) / w[1]

    plt.plot(xx, yy, '-r', label='Hiperplano de decisión')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()

# EJEMPLO 2:
if True:
    # Datos
    X = np.array([[2, 2], [2, 3], [3, 3], [3, 4], [4, 4], [4, 5]])
    y = np.array([1, 1, 1, -1, -1, -1])

    # Crear y entrenar el modelo SVM
    svm = SoftMarginSVM(C=1.0, max_iter=100, tol=1e-3)
    svm.fit(X, y)

    # Dibujar los puntos de datos
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    # Dibujar el hiperplano de decisión
    ax = plt.gca()
    xlim = ax.get_xlim()
    w = (svm.alphas * y).dot(X)
    b = svm.b
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = (-w[0] * xx - b) / w[1]

    plt.plot(xx, yy, '-r', label='Hiperplano de decisión')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()

    

# EJEMPLO 3:
if True:
    # Datos
    X = np.array([[1, 1], [2, 2], [2, 3], [4, 4], [5, 5], [5, 6]])
    y = np.array([1, -1, 1, 1, -1, -1])

    # Crear y entrenar el modelo SVM
    svm = SoftMarginSVM(C=1.0, max_iter=100, tol=1e-3)
    svm.fit(X, y)

    # Dibujar los puntos de datos
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    # Dibujar el hiperplano de decisión
    ax = plt.gca()
    xlim = ax.get_xlim()
    w = (svm.alphas * y).dot(X)
    b = svm.b
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = (-w[0] * xx - b) / w[1]

    plt.plot(xx, yy, '-r', label='Hiperplano de decisión')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()

# Ejemplo con 30 puntos
if True:
    # Generar un conjunto de datos aleatorio en 3D
    np.random.seed(0)
    X = np.random.randn(100, 3)
    y = np.where(X[:, 0] + X[:, 1] - X[:, 2] > 0, 1, -1)

    # Colores para los puntos
    colors = np.where(y == 1, 'b', 'r')  # Azul para la clase 1, Rojo para la clase -1

    # Crear y entrenar el modelo SVM
    svm = SoftMarginSVM(C=1.0, max_iter=100, tol=1e-3)
    svm.fit(X, y)

    # Crear una figura en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Dibujar los puntos de datos con colores
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors)

    # Definir el rango para el gráfico
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 30), np.linspace(ylim[0], ylim[1], 30))

    # Calcular los valores en el plano
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    zz = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = svm.decision_function(np.append(grid_points[i * xx.shape[1] + j], 0))

    # Dibujar el hiperplano de decisión
    ax.plot_surface(xx, yy, zz, color='green', alpha=0.3)  # Cambiamos el color del hiperplano a verde
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    plt.show()