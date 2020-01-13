import numpy as np

def get_A(epsilon):
    ret = np.empty((4, 3))
    ret[0, :] = 1
    ret[1:, :] = np.eye(3) * epsilon
    return ret


def compare_implementations(epsilon):
    A = get_A(epsilon)

    print("naive implementation")
    C = np.matmul(A.T, A)
    print("C:")
    print(C.astype(str))
    eigenvals = np.linalg.eigvals(C)
    print("Eigenvalues:")
    print(eigenvals.astype(str))
    s = np.sqrt(eigenvals)
    print("Singular values:")
    print(s.astype(str))

    print("-------------------")

    print("Proper implementation")
    _, s, _ = np.linalg.svd(A)
    print("Singular values:")
    print(s.astype(str))
    print("#######################")
    print()


def main():
    print("Epsilon = 1e-7")
    compare_implementations(1e-7)
    print("Epsilon = 1e-8")
    compare_implementations(1e-8)

if __name__ == '__main__':
    main()
