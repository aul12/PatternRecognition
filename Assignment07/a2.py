import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sklearn.metrics


def calc_scatter(data, labels):
    _, class_counts = np.unique(labels, return_counts=True)
    n_classes = len(class_counts)
    f_vector_len = data.shape[-1]
    sorted_samples = []
    for c in range(n_classes):
        sorted_samples.append(np.empty((class_counts[c], f_vector_len)))

    class_counters = np.zeros((n_classes, ), dtype=np.int16)
    for c in range(data.shape[0]):
        index = labels[c] - 1
        sorted_samples[index][class_counters[index], :] = data[c, :]
        class_counters[index] += 1

    class_means = np.empty((n_classes, f_vector_len))
    class_variances = np.empty((n_classes, f_vector_len, f_vector_len))
    for class_index in range(len(sorted_samples)):
        class_means[class_index, :] = np.mean(sorted_samples[class_index], axis=0)
        class_variances[class_index, :, :] = np.cov(sorted_samples[class_index], rowvar=False)

    global_mean = np.mean(class_means, axis=0)
    scatter_within = np.sum(class_variances, axis=0)
    scatter_between = np.zeros((f_vector_len, f_vector_len))
    for c in range(n_classes):
        scatter_between += class_counts[c] * np.outer(class_means[c] - global_mean, class_means[c] - global_mean)

    return scatter_within, scatter_between


def compute_lda(data, labels):
    s_w, s_b = calc_scatter(data, labels)
    s_product = np.matmul(np.linalg.pinv(s_w),  s_b)
    eigenvalues, eigenvectors = np.linalg.eig(s_product)
    sorted_indices = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, sorted_indices]
    projection = np.matmul(data, eigenvectors)
    return eigenvectors, projection, eigenvalues[sorted_indices]


def plot_2D_space(data, colormap, title=None, save_as=None):
    assert data.shape[-1] == 2 and len(data.shape) == 2
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=colormap)
    if title:
        plt.title(title)

    if save_as:
        plt.savefig(save_as)

    plt.show()


def plot_surfaces(data, classifier):
    max_x = np.max(data[:, 0])
    min_x = np.min(data[:, 0])
    max_y = np.max(data[:, 1])
    min_y = np.min(data[:, 1])
    x, y = np.meshgrid(np.arange(min_x, max_x, 0.1), np.arange(min_y, max_y, 0.1))
    print("test")


def main():
    print("loading data...")
    raw = np.array(pd.read_csv("train.csv"))
    data = raw[:, 0:-1]
    labels = raw[:, -1]
    print("computing lda...")
    axes, projection, latent = compute_lda(data, labels)
    abs_latent = np.abs(latent)
    importance = abs_latent / np.sum(abs_latent)
    for c in range(len(importance)):
        print("eigenvalue %d has importance %0.3f" % (c, importance[c]))

    reduced_space = projection[:, -2:]
    reduced_space = np.real(reduced_space) + np.imag(reduced_space)
    plot_surfaces(reduced_space, reduced_space)
    classifier = SVC()
    print("training svm...")
    classifier.fit(reduced_space, labels)
    pred_labels = classifier.predict(reduced_space)
    accuracy_score = sklearn.metrics.accuracy_score(labels, pred_labels)
    print("Accuracy score: %0.3f" % (accuracy_score, ))
    colormap = {
        1: "blue",
        2: "red",
        3: "green",
        4: "black"
    }
    plot_2D_space(reduced_space, list(map(colormap.get, labels)), "Ground truth")
    plot_2D_space(reduced_space, list(map(colormap.get, pred_labels)), "Predicted classes")
    print("done")


if __name__ == '__main__':
    main()
