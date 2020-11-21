import numpy as np


def split_matrix(A):
    D = np.zeros_like(A)
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    for i in range(A.shape[0]):
        D[i, i] = A[i, i]

    for i in range(A.shape[0]):
        for j in range(i):
            L[i, j] = A[i, j]

    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            U[i, j] = A[i, j]
    print("D:", D)
    print("L:", L)
    print("U:", U)
    return D, L, U


def execute(A, b):
    assert A.shape[0] == A.shape[1], "A must be n * n"
    assert A.shape[0] == b.shape[0], "size of A and b must match"
    print("A:", A)
    print("b:", b)
    # D, L, U = split_matrix(A)
    x_prev = np.zeros_like(b)
    x = np.zeros_like(b)
    w = 0.7
    while True:
        x_prev = np.copy(x)
        for i in range(A.shape[0]):
            mida = 0
            midb = 0
            for j in range(i):
                mida += A[i, j] * x[j]
            for j in range(i + 1, A.shape[0]):
                midb += A[i, j] * x_prev[j]
            x[i] = (
                -w * mida + (1 - w) * A[i, i] * x_prev[i] - w * midb + w * b[i]
            ) / A[i, i]
        if (np.abs(x - x_prev) < 0.001).all():
            break
    return x


def single(A, b):
    print("-----single-----")
    A_copy = np.copy(A).astype(np.float32)
    b_copy = np.copy(b).astype(np.float32)
    x = execute(A_copy, b_copy)
    print("A * x:", A @ x.T)
    print("-----END-----")


def double(A, b):
    print("-----single-----")
    A_copy = np.copy(A).astype(np.float64)
    b_copy = np.copy(b).astype(np.float64)
    x = execute(A_copy, b_copy)
    print("A * x:", A @ x.T)
    print("-----END-----")


if __name__ == "__main__":
    A = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 2.0],
        ],
    )
    b = np.array([1.0, 0.0, 0.0, 0.0])
    double(A, b)
