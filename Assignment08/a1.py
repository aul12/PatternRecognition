import numpy as np
from sklearn.svm import LinearSVC


def phi(x):
    return np.array([x, (x+0.5)**2])


def main():
    data_points = [-2, -1, 0, 2]
    classes = [1, -1, -1, 1]
    # 1.1
    projected_data_points = list(map(phi, data_points))
    # 1.2
    svm = LinearSVC()
    svm.fit(projected_data_points, classes)
    print("Decision hyper plane defined by w = (%0.3f, %0.3f) and w_0 = %0.3f" %
          (svm.coef_[0, 0], svm.coef_[0, 1], svm.intercept_))
    # 1.3
    x_5 = 1
    projected_x_5 = np.reshape(phi(x_5), (-1, 1))
    classification = np.sign(np.dot(svm.coef_, projected_x_5) + svm.intercept_).astype(np.int)
    print("Manual classification yields: " + str(classification[0]))
    print("Classification by svm object yields: " + str(svm.predict(projected_x_5.T)))


if __name__ == "__main__":
    main()
