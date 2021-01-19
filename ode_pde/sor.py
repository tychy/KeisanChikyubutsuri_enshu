import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def execute(A, b, w):
    assert A.shape[0] == A.shape[1], "A must be n * n"
    assert A.shape[0] == b.shape[0], "size of A and b must match"
    x_prev = np.zeros_like(b)
    x = np.zeros_like(b)
    step = 0
    max_step = 300
    while step <= max_step:
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
        step += 1
    return x
