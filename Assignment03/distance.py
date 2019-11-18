import numpy as np
import math

# a)
def distance(xi, xj, matrixA):
    diff = np.expand_dims(
            np.array(xi) - np.array(xj), axis=1)
    diffT = np.transpose(diff)
    dSquare = np.matmul(
            diffT, np.matmul(np.linalg.inv(matrixA), diff))
    return math.sqrt(dSquare)

# b)
data = np.transpose([[1,1200],
        [1.5,1000],
        [1.9,1300],
        [3,1900],
        [5,3200],
        [5.5,2400],
        [7,4200]])

cov = np.cov(data)

A_euler = np.eye(2)
A_normalized = np.array([[cov[0][0], 0],
                         [0, cov[1][1]]])
A_mahalanobis = cov

# c)
print("Euclidean: ||(7,4200) - (5,3200)|| = %.2f"
        % distance([7,4200], [5, 3200], A_euler))
print("Standardization: ||(7,4200) - (5,3200)|| = %.2f"
        % distance([7,4200], [5, 3200], A_normalized))
print("Mahalanobis: ||(1.5,1000) - (1.9,1300)|| = %.2f"
        % distance([1.5,1000], [1.9,1300], A_mahalanobis))
print("Mahalanobis: ||(1.9,1300) - (1.5,1000)|| = %.2f"
        % distance([1.9,1300], [1.5,1000], A_mahalanobis))
