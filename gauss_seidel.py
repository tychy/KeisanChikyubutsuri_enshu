import numpy as np


def scale(A, x):
    scale_ls = np.zeros_like(x)
    for i in range(A.shape[0]):
        max_idx = 0
        for j in range(A.shape[0]):
            if np.abs(A[i, j]) > np.abs(A[i, max_idx]):
                max_idx = j
        scale_ls[i] = np.abs(A[i, max_idx])
        x[i] = x[i] / np.abs(A[i, max_idx])
        A[i, :] = A[i, :] / np.abs(A[i, max_idx])
    return scale_ls


def pivot(A, x, idx):
    max_idx = idx
    for i in range(idx, A.shape[0]):
        if A[idx, i] > A[idx, max_idx]:
            max_idx = i
    if max_idx != idx:
        buff = np.copy(A[max_idx, :])
        A[max_idx, :] = A[idx, :]
        A[idx, :] = buff
        buffx = x[max_idx]
        x[max_idx] = x[idx]
        x[idx] = buffx
    return


def execute(A, b):
    assert A.shape[0] == A.shape[1], "A must be n * n"
    assert A.shape[0] == b.shape[0], "size of A and b must match"
    print("A:", A)
    print("b:", b)
    L = np.zeros_like(A)
    for i in range(A.shape[0]):
        L[i, i] = 1
    scale_ls = scale(A, b)
    print("Scaled A:", A)

    pivot(A, b, 0)
    for i in range(1, A.shape[0]):
        pivot(A, b, i)
        for j in range(i, A.shape[0]):
            coef = A[j][i - 1] / A[i - 1][i - 1]
            A[j, :] = A[j, :] - A[i - 1, :] * coef
            b[j] = b[j] - b[i - 1] * coef
            L[j, i - 1] = coef
    x = np.zeros_like(b)
    print("L:", L)
    for i in range(A.shape[0] - 1, -1, -1):
        x[i] += b[i]
        for j in range(i, A.shape[0] - 1):
            x[i] -= A[i][j + 1] * x[j + 1]
        x[i] /= A[i][i]
    print("x:", x)
    for i in range(A.shape[0]):
        A[i, :] = A[i, :] * scale_ls[i]

    return x, L, A


def double(A, b):
    print("-----single-----")
    A_copy = np.copy(A).astype(np.float64)
    b_copy = np.copy(b).astype(np.float64)
    x, L, U = execute(A_copy, b_copy)
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

