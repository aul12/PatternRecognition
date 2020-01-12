import pickle
import sys
import numpy as np
from matplotlib import pyplot as plt


# a 1.1
def plot_image(data, title=None):
    max_loc = np.max(data)
    min_loc = np.min(data)
    plt.figure()
    plt.xlim(min_loc, max_loc)
    plt.ylim(min_loc, max_loc)
    plt.scatter(data[:, 0], data[:, 1])
    if title:
        plt.title(title)

    plt.show()


def main():
    data_path = sys.argv[1]
    data = pickle.load(open(data_path, "rb"))
    n_samples = max(data.shape)
    plot_image(data, "Original")

    # a 1.2
    means = np.mean(data, axis=0)
    data_centr = data - means

    # a 1.3
    covariance_matrix = np.matmul(data_centr.T, data_centr) / (n_samples - 1)
    print(covariance_matrix.shape)

    # a 1.4 a)
    us_eval, us_evec = np.linalg.eig(covariance_matrix)
    sorted_indices = list(reversed(np.argsort(us_eval)))
    eval = us_eval[sorted_indices]
    evec = us_evec[:, sorted_indices]

    # a 1.4 b)
    var_total = np.sum(eval)
    print("Explained variance along axis 1 is %0.3f%%" % (eval[0] / var_total))
    print("Explained variance along axis 2 is %0.3f%%" % (eval[1] / var_total))

    # a 1.5
    data_proj = np.matmul(evec, data_centr.T)
    plot_image(data_proj.T, "Projected data")


if __name__ == '__main__':
    main()
