import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import lda


def main():
    sum_len = 18
    # a 3.1
    data_path = sys.argv[1]
    data, labels = lda.load_data(data_path)

    # a 3.2
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # a 3.3
    pca = PCA(n_components=2)
    projections = pca.fit_transform(data_scaled)

    # a 3.4
    pca = PCA(n_components=sum_len)
    pca.fit(data_scaled)
    cul_sum = np.zeros(sum_len)
    cul_sum[0] = pca.explained_variance_[0]
    total_var = np.sum(pca.explained_variance_)
    for i in range(1, sum_len):
        cul_sum[i] = cul_sum[i - 1] + pca.explained_variance_[i]

    cul_sum /= total_var
    plt.figure()
    plt.xticks(np.arange(1, sum_len + 1, 1))
    plt.xlabel("Number of eigenvectors")
    plt.ylabel("Explained varince in %")
    plt.title("Explained variance (culmulated)")
    plt.grid(True)
    plt.plot(np.arange(1, sum_len + 1, 1), cul_sum, ".-")
    plt.show()

    # a 3.5
    legend = ["Opel", "Saab", "Bus", "Van"]
    _, lda_projection, _ = lda.compute_lda(data_scaled, labels)
    reduced_space = lda_projection[:, -2:]
    reduced_space = np.real(reduced_space) + np.imag(reduced_space)
    lda.plot_scatter(projections, labels, legend=legend, title="Projection by PCA")
    lda.plot_scatter(reduced_space, labels, legend=legend, title="Projection by LDA")
    plt.show()


if __name__ == '__main__':
    main()
