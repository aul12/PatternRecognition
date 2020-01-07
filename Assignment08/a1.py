import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


def main():
    data_points = [-2, -1, 0, 2]
    classes = [1, -1, -1, 1]
    # 1.1
    def phi(x):
        return np.array([x, (x+0.5)**2])

    projected_data_points = list(map(phi, data_points))
    # 1.2
    svm = LinearSVC()
    svm.fit(projected_data_points, classes)
    # the linear SVC object only returns the decision hyper plane not the alpha parameters...
    print("Decision hyper plane defined by w = (%0.3f, %0.3f) and w_0 = %0.3f" %
          (svm.coef_[0, 0], svm.coef_[0, 1], svm.intercept_))
    # 1.3
    x_5 = 1
    projected_x_5 = np.reshape(phi(x_5), (-1, 1))
    classification = np.sign(np.dot(svm.coef_, projected_x_5) + svm.intercept_).astype(np.int)
    # alphas computed manually
    alpha = [1, 1, 0, 0]
    print("Manual classification yields: " + str(classification[0]))
    print("Classification by svm object yields: " + str(svm.predict(projected_x_5.T)))

    # 1.4
    def k(x_i, x):
        return np.power(x_i, 2) * (np.power(x, 2) + x + 0.25) \
                + x_i * (2 * np.power(x, 2) + 1.5*x + 0.25) \
                + 0.25 * np.power(x, 2) + 0.25*x + 0.0625

    print("== Kernel ==")
    kernel_classification = np.sign(np.sum(np.array(alpha) * np.array(classes) * k(x_5, np.array(data_points))))
    print("Classification yields: " + str(kernel_classification))

    # 1.5
    def gram_kernel(data):
        N = len(data)
        gram = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                gram[i][j] = k(data[i], data[j])

        return gram

    kernel_svm = SVC(kernel="precomputed")
    kernel_svm.fit(gram_kernel(data_points), classes)

    # 1.6
    def dist(x):
        unnormalized = np.sum(np.array(alpha) * np.array(classes) * k(x_5, np.array(data_points)))
        squared_norm = 0
        for i in range(4):
            for j in range(4):
                squared_norm += classes[i] * classes[j] * alpha[i] * alpha[j] * k(data_points[i], data_points[j])

        return unnormalized / np.sqrt(squared_norm)

    print("Distance to hyper plane: %f" % dist(x_5))

if __name__ == "__main__":
    main()
