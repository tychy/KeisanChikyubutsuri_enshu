import numpy as np


def execute(A, b):
    assert A.shape[0] == A.shape[1], "A must be n * n"
    assert A.shape[0] == b.shape[0], "size of A and b must match"
    for i in range(1, A.shape[0]):
        for j in range(i, A.shape[0]):
            coef = A[j][i - 1] / A[i - 1][i - 1]
            A[j, :] = A[j, :] - A[i - 1, :] * coef
            b[j] = b[j] - b[i - 1] * coef
    x = np.zeros_like(b)
    for i in range(A.shape[0] - 1, -1, -1):
        x[i] += b[i]
        for j in range(i, A.shape[0] - 1):
            x[i] -= A[i][j + 1] * x[j + 1]
        x[i] /= A[i][i]
    return x


if __name__ == "__main__":
    A = np.array(
        [
            [10.0, 1.0, 4.0, 0.0],
            [1.0, 10.0, 5.0, -1.0],
            [4.0, 5.0, 10.0, 7.0],
            [0.0, -1.0, 7.0, 9.0],
        ],
        dtype=np.float64,
    )
    A_copy = np.copy(A)
    b = np.array([16.0, 16.0, 25.0, 16.0], dtype=np.float64)
    x = execute(A_copy, b)
    print(x.dtype)
    print(A_copy.dtype)
    print(A @ x.T)
