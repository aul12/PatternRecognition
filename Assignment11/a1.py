import scipy.io.wavfile
import sys
import numpy as np
import sklearn.decomposition
import matplotlib.pyplot as plt


# 4 a)
def getentropy(values):
    assert len(values.shape) == 1
    hist, _ = np.histogram(values, bins=500)
    p = hist / np.sum(hist)
    log_val = np.log2(p)
    # convergence for the win!
    log_val[np.isinf(log_val)] = 0
    return - np.sum(np.multiply(p, log_val))


def get_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def plot_scatter(data, title=None, save_as=None, legend=None):
    """4. d) Create a scatter plot of a 2D dataset"""
    assert data.shape[-1] == 2 and len(data.shape) == 2
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("Mic 1")
    plt.ylabel("Mic 2")

    if title:
        plt.title(title)

    if legend:
        plt.legend()

    if save_as:
        plt.savefig(save_as)

    plt.show()


def main():
    if len(sys.argv) != 3:
        print("Please specify the .wav files as program arguments")
        exit(1)

    # 1 load data
    sample_rate_A, signal_A = scipy.io.wavfile.read(sys.argv[1])
    sample_rate_B, signal_B = scipy.io.wavfile.read(sys.argv[2])
    data_mixed = np.stack((signal_A, signal_B), axis=-1)
    plot_scatter(data_mixed, title="Mixed data X")

    # 2 calculate centralized data
    data_centr = data_mixed - np.mean(data_mixed, axis=0)
    plot_scatter(data_centr, title="Centered data X_c")

    # 3 whitening
    pca = sklearn.decomposition.PCA(n_components=2, whiten=True)
    data_whitened = pca.fit_transform(data_centr)
    plot_scatter(data_whitened, title="Whitened data X_w")

    # 4 b)
    thetas = np.arange(0, np.pi, np.pi / 180)
    rot_mats = np.rollaxis(get_rot_mat(thetas), -1)
    transformations = np.matmul(data_whitened, rot_mats)
    mi_values = np.empty(transformations.shape[0])
    for i in range(mi_values.shape[0]):
        mi_values[i] = getentropy(transformations[i, :, 0])
        mi_values[i] += getentropy(transformations[i, :, 1])

    min_index = np.argmin(mi_values)
    theta_min = min_index * np.pi / 180
    plt.figure()
    plt.plot(thetas, mi_values)
    plt.plot((theta_min, theta_min), (np.min(mi_values), np.max(mi_values)))
    plt.title("Find U transposed")
    plt.xlabel("theta in radiants")
    plt.ylabel("Multi-Information")
    plt.legend(["mi(theta)", "optimal theta"])
    plt.show()

    print("Optimal theta: %0.1f°" % (theta_min * 180 / np.pi))
    print("Multi-Information at optimum: %0.2f" % (mi_values[min_index]))

    # 5 a)
    data_reconstructed = np.matmul(data_whitened, rot_mats[min_index])

    # 5 b)
    data_reconstructed -= np.min(data_reconstructed, axis=0)
    data_reconstructed /= (np.max(data_reconstructed, axis=0) - np.min(data_reconstructed, axis=0))
    plot_scatter(data_reconstructed, title="Reconstructed data X_r")

    time_axis = np.arange(0, data_reconstructed.shape[0])
    plt.figure()
    plt.plot(time_axis / sample_rate_A, data_reconstructed[:, 0])
    plt.title("Reconstructed source 1")
    plt.xlabel("Time in s")
    plt.ylabel("Amplitude")
    plt.show()

    plt.figure()
    plt.plot(time_axis / sample_rate_B, data_reconstructed[:, 1])
    plt.title("Reconstructed source 2")
    plt.xlabel("Time in s")
    plt.ylabel("Amplitude")
    plt.show()


if __name__ == '__main__':
    main()

