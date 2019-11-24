import matplotlib.pyplot as plt
import numpy as np

def L(p):
    return p**3 * (1-p)**5

def main():
    x = np.arange(0, 1, 0.01)
    y = np.vectorize(L)(x)

    fig = plt.figure()
    plt.plot(x, y)
    plt.plot([3/8, 3/8], [0,L(3/8)])
    plt.xlabel("p")
    plt.ylabel("L(p)")
    plt.title("Likelihood function")
    plt.show()
    fig.savefig("A1.eps")


if __name__ == "__main__":
    main()
