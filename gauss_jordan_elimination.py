import numpy as np


def pivot(A, idx):
    max_idx = idx
    for i in range(idx, A.shape[0]):
        if A[idx, i] > A[idx, max_idx]:
            max_idx = i
    if max_idx != idx:
        buff = np.copy(A[:, max_idx])
        A[:, max_idx] = A[:, idx]
        A[:, idx] = buff
    return


def execute(A, b):
    assert A.shape[0] == A.shape[1], "A must be n * n"
    assert A.shape[0] == b.shape[0], "size of A and b must match"
    print("A:", A)
    print("b:", b)
    pivot(A, 0)
    for i in range(1, A.shape[0]):
        pivot(A, i)
        for j in range(i, A.shape[0]):
            coef = A[j][i - 1] / A[i - 1][i - 1]
            A[j, :] = A[j, :] - A[i - 1, :] * coef
            b[j] = b[j] - b[i - 1] * coef
    x = np.zeros_like(b)
    print(A)
    for i in range(A.shape[0] - 1, -1, -1):
        x[i] += b[i]
        for j in range(i, A.shape[0] - 1):
            x[i] -= A[i][j + 1] * x[j + 1]
        x[i] /= A[i][i]
    print("x:", x)
    return x


def single(A, b):
    print("-----single-----")
    A_copy = np.copy(A).astype(np.float32)
    b_copy = np.copy(b).astype(np.float32)
    x = execute(A_copy, b_copy)
    print("A * x:", A @ x.T)
    print("-----END-----")


def double(A, b):
    print("double")

    A_copy = np.copy(A).astype(np.float64)
    b_copy = np.copy(b).astype(np.float64)
    x = execute(A_copy, b_copy)
    print("A * x:", A @ x.T)
    print("-----END-----")


if __name__ == "__main__":
    A = np.array(
        [
            [10.0, 1.0, 4.0, 0.0],
            [1.0, 10.0, 5.0, -1.0],
            [4.0, 5.0, 10.0, 7.0],
            [0.0, -1.0, 7.0, 9.0],
        ],
    )
    b = np.array([15.0, 15.0, 25.0, 15.0])
    c = np.array([16.0, 16.0, 25.0, 16.0])

    single(A, b)
    double(A, b)
    single(A, c)
    double(A, c)
