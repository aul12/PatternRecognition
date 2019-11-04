import matplotlib.pyplot as plt
import numpy as np
import sys


def likelihood_function(mean):
    def p(x):
        return 1 / np.sqrt(np.pi) * np.exp(-(x - mean) ** 2)

    return p


def loss(lamb, likelihood, prior):
    def l(x):
        return lamb * likelihood(x) * prior

    return l


def calc_thresh(lamb12, lamb21, p):
    return 0.5 * (np.log(lamb12 / lamb21 * p / (1-p)) + 1)


def plot_losses(lamb12, lamb21, lik1, lik2, p, x, name, show=False):
    plt.figure()
    l1 = loss(lamb12, lik1, p)(x)
    l2 = loss(lamb21, lik2, 1 - p)(x)
    plt.plot(x, l1)
    plt.plot(x, l2)
    thresh = calc_thresh(lamb12, lamb21, p)
    plt.plot([thresh, thresh], [min(np.min(l1), np.min(l2)), max(np.max(l1), np.max(l2))], "k-", lw=2)
    plt.legend(["l_1", "l_2", "Threshold"])
    plt.title("Losses for lambda_12 = %1.2f, lambda_21 = %1.2f, p = %1.2f" % (lamb12, lamb21, p))
    plt.xlabel("x")
    plt.ylabel("l(x)")
    plt.savefig(name)
    if show:
        plt.show()


def main():
    show = len(sys.argv) > 1
    x = np.arange(-3, 3, 0.01)
    mean1 = 0
    mean2 = 1
    lambs12 = [0.5, 7/2, 7]
    lambs21 = [0.5, 1, 2]
    ps = [0.5, 1/3, 1/3]
    plt.figure()
    lik1 = likelihood_function(mean1)
    lik2 = likelihood_function(mean2)
    plt.plot(x, lik1(x))
    plt.plot(x, lik2(x))
    plt.plot([0.5, 0.5], [0, 0.6], "k-", lw=2)
    plt.legend(["w_1", "w_2", "Threshold"])
    plt.title("Likelihood functions")
    plt.xlabel("x")
    plt.ylabel("p(x|w)")
    if show:
        plt.show()
    plt.savefig("A24_Likelihoods.eps")
    ex_no = 4
    for lamb12, lamb21, p in zip(lambs12, lambs21, ps):
        plot_losses(lamb12, lamb21, lik1, lik2, p, x, "A2%d_Losses.eps" % (ex_no, ), show)
        ex_no += 1

    plot_losses(1, 1, lik1, lik2, 1/3, x, "A27a_Losses.eps", show)
    plot_losses(1, 1, lik1, lik2, 1/2, x, "A27b_Losses.eps", show)
    plot_losses(1, 1, lik1, lik2, 2/3, x, "A27c_Losses.eps", show)


if __name__ == '__main__':
    main()
