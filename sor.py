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


def execute(A, b, w, x_true):
    assert A.shape[0] == A.shape[1], "A must be n * n"
    assert A.shape[0] == b.shape[0], "size of A and b must match"
    print("A:", A)
    print("b:", b)
    # D, L, U = split_matrix(A)
    x_prev = np.zeros_like(b)
    x = np.zeros_like(b)
    step_ls = []
    x_ls = []
    step = 0
    max_step = 300
    while step <= max_step:
        if np.linalg.norm(x - x_true) > np.power(10, 10):
            break
        if step % 10 == 0:
            step_ls.append(step)
            x_ls.append(np.log(np.linalg.norm(x - x_true)))
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
    print("x:", x)
    return x, step_ls, x_ls


def double(A, b, w, x_true):
    print("-----double-----")
    print("omega:", w)
    A_copy = np.copy(A).astype(np.float64)
    b_copy = np.copy(b).astype(np.float64)
    x, step_ls, x_ls = execute(A_copy, b_copy, w, x_true)
    print("A * x:", A @ x.T)
    print("-----END-----")
    return step_ls, x_ls


def plot_sor():
    sns.set_theme()
    A = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 2.0],
        ],
    )
    b = np.array([1.0, 0.0, 0.0, 0.0])
    x_true = np.array(
        [0.72727272727273, -0.27272727272727, -0.18181818181818, -0.090909090909091]
    )
    fig = plt.figure(figsize=(10, 10))
    for w in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0, 1.01, 1.1, 1.3, 1.5, 1.7, 1.9, 3, 5]:
        step_ls, x_ls = double(A, b, w, x_true)
        plt.plot(step_ls, x_ls, label="w:{:.2f}".format(w))

    plt.xlabel("step")
    plt.ylabel("ln|x-x_t|")
    plt.legend()
    plt.savefig("sor.png")


def plot_sor_debug():
    sns.set_theme()
    A = np.array(
        [
            [10.0, 1.0, 4.0, 0.0],
            [1.0, 10.0, 5.0, -1.0],
            [4.0, 5.0, 10.0, 7.0],
            [0.0, -1.0, 7.0, 9.0],
        ],
    )
    b = np.array([15.0, 15.0, 26.0, 15.0])

    x_true = np.array([1, 1, 1, 1])
    fig = plt.figure(figsize=(10, 10))

    for w in [0.1, 0.7, 1.0, 1.1, 1.5, 1.7, 1.8, 1.9]:
        step_ls, x_ls = double(A, b, w, x_true)
        plt.plot(step_ls, x_ls, label="w:{:.1f}".format(w))

    plt.xlabel("step")
    plt.ylabel("ln|x-x_t|")
    plt.legend()
    plt.savefig("sor_debug.png")


if __name__ == "__main__":
    plot_sor()
    plot_sor_debug()