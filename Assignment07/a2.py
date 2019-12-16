import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sklearn.metrics
import sys


def sort_classes(data, labels):
    """Splits the data into separate data blocks according to the specified labels"""
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

    return n_classes, class_counts, f_vector_len, sorted_samples


def calc_scatter(data, labels):
    """2. a) Calculates both scatter matrices"""
    n_classes, class_counts, f_vector_len, sorted_samples = sort_classes(data, labels)
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
    """2. b) - d) Computes the LDA and the resulting projection"""
    s_w, s_b = calc_scatter(data, labels)
    s_product = np.matmul(np.linalg.pinv(s_w),  s_b)
    eigenvalues, eigenvectors = np.linalg.eig(s_product)
    sorted_indices = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, sorted_indices]
    projection = np.matmul(data, eigenvectors)
    return eigenvectors, projection, eigenvalues[sorted_indices]


def plot_scatter(data, labels, title=None, save_as=None, legend=None):
    """4. d) Create a scatter plot of a 2D dataset"""
    assert data.shape[-1] == 2 and len(data.shape) == 2
    n_classes, _, _, sorted_classes = sort_classes(data, labels)
    assert not legend or n_classes == len(legend)
    plt.figure()
    for c in range(n_classes):
        plt.scatter(sorted_classes[c][:, 0], sorted_classes[c][:, 1], label=legend[c])

    if title:
        plt.title(title)

    if legend:
        plt.legend()

    if save_as:
        plt.savefig(save_as)

    # plt.show()


def plot_surfaces(data, classifier):
    """4. d) Add separation lines to scatter plot"""
    max_x = np.max(data[:, 0])
    min_x = np.min(data[:, 0])
    max_y = np.max(data[:, 1])
    min_y = np.min(data[:, 1])
    x, y = np.meshgrid(np.arange(min_x, max_x, 0.1), np.arange(min_y, max_y, 0.1))
    in_points = np.stack((np.ndarray.flatten(x), np.ndarray.flatten(y)), axis=-1)
    res = classifier.predict(in_points)
    res_surface = np.reshape(res, x.shape)
    plt.contour(x, y, res_surface)


def load_data(path):
    """3. Loads a cvs file and splits it into data and labels"""
    print("loading data from %s..." % (path, ))
    raw = np.array(pd.read_csv("train.csv"))
    data = raw[:, 0:-1]
    labels = raw[:, -1]
    return data, labels


def main():
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    data, labels = load_data(train_data_path)
    print("computing lda...")
    axes, projection, latent = compute_lda(data, labels)
    # 4. a) Compute importance of axis
    abs_latent = np.abs(latent)
    importance = abs_latent / np.sum(abs_latent)
    for c in range(len(importance)):
        print("eigenvalue %d has importance %0.3f" % (c, importance[c]))

    # 4. b) Take only the two most important dimensions of the projection
    reduced_space = projection[:, -2:]
    reduced_space = np.real(reduced_space) + np.imag(reduced_space)

    # 4. c) Train the SVM
    classifier = SVC()
    print("training svm...")
    classifier.fit(reduced_space, labels)
    pred_labels = classifier.predict(reduced_space)
    accuracy_score = sklearn.metrics.accuracy_score(labels, pred_labels)
    print("Accuracy score: %0.3f" % (accuracy_score, ))
    plot_scatter(reduced_space, labels, title="Ground truth train data",
                 legend=["Opel", "Saab", "Bus", "Van"])
    plot_scatter(reduced_space, pred_labels, title="Predicted classes on train data",
                 legend=["Opel", "Saab", "Bus", "Van"])
    plot_surfaces(reduced_space, classifier)
    plt.show()
    print("training done")

    # 5. Do all the stuff again (for a test set which seems to be the same as the train set, at least on December the 16th...)
    test_data, test_labels = load_data(test_data_path)
    print("projecting test data...")
    projected_test_data = np.matmul(test_data, axes)[:, -2:]
    projected_test_data = np.real(projected_test_data) + np.imag(projected_test_data)
    print("classifying test data...")
    pred_labels_test = classifier.predict(projected_test_data)
    c_matrix = sklearn.metrics.confusion_matrix(test_labels, pred_labels_test)
    accuracy_score = sklearn.metrics.accuracy_score(labels, pred_labels)
    print("Accuracy score: %0.3f" % (accuracy_score, ))
    print("Confusion matrix:")
    print(c_matrix)
    plot_scatter(projected_test_data, pred_labels_test, title="Predicted classes on test data",
                 legend=["Opel", "Saab", "Bus", "Van"])
    plot_surfaces(projected_test_data, classifier)
    plt.show()


if __name__ == '__main__':
    main()
