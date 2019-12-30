import numpy as np
from sklearn.svm import LinearSVC

def phi(x):
    return np.array([x, (x+0.5)**2])

def main():
    data_points = [-2, -1, 0, 2]
    classes = [1, -1, -1, 1]
    projected_data_points = list(map(phi, data_points))
    svm = LinearSVC()
    svm.fit(projected_data_points, classes)
    print(svm.coef_)

if __name__ == "__main__":
    main()
