import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time


def gen_data(n, case):
    if case == 1:
        arr = np.random.rand(n)
    else:
        arr = []
        theta = np.linspace(0, 2 * np.pi, n)
        for item in theta:
            if item <= np.pi:
                value = -1 + 2 * item / np.pi
            else:
                value = 3 - 2 * item / np.pi
            arr.append(value)
        arr = np.array(arr)

    return arr


def coef(f_ls):
    size = len(f_ls)
    coef_ls = []
    for i in range(size):
        c_k = 0
        for idx, item in enumerate(f_ls):
            c_k += item * np.exp(-1.0j * 2 * np.pi * i * idx / size)
        coef_ls.append(c_k / size)
    coef_ls = np.array(coef_ls)
    return coef_ls


def dft(coef_ls):
    size = len(coef_ls)
    f_ls = []
    for i in range(size):
        f_n = 0
        for idx, item in enumerate(coef_ls):
            f_n += coef_ls[idx] * np.exp(1.0j * 2 * np.pi * i * idx / size)
        f_ls.append(np.real(f_n))
    return f_ls


def test_dft(n, case):
    if case == 1:
        arr = gen_data(n, case)
        coef_ls = coef(arr)
        f_ls = dft(coef_ls)
        print("input:", arr)
        print("fourier:", f_ls)
    else:
        sns.set_theme()
        time_ls = []
        error_ls = []
        step_ls = []

        for idx in tqdm(range(1, n)):
            arr = gen_data(idx, case)
            ut = time.time()
            coef_ls = coef(arr)
            t = time.time() - ut

            f_ls = dft(coef_ls)

            c_true = np.zeros_like(coef_ls)
            for i in range(1, idx):
                c_true[i] = 2 / np.pi / np.pi / i / i * ((-1) ** i - 1)
            diff = coef_ls - c_true
            error_ls.append(np.mean(np.sqrt(diff.real ** 2 + diff.imag ** 2)))
            step_ls.append(idx)
            time_ls.append(t)

        # draw
        step_ls = np.array(step_ls)
        error_ls = np.array(error_ls)
        figure = plt.figure()
        ax1 = figure.add_subplot(1, 2, 1)
        ax1.set_xlabel("N")
        ax1.set_ylabel("f(t) - f_N(t)")
        ax1.plot(step_ls, error_ls)

        ax2 = figure.add_subplot(1, 2, 2)
        ax2.plot(np.log(step_ls), np.log(error_ls))
        ax2.set_xlabel("log(N)")
        ax2.set_ylabel("log(f(t) - f_N(t))")
        figure.tight_layout()
        figure.savefig("dft_error.png")

        # draw
        time_ls = np.array(time_ls)
        figure = plt.figure()
        ax1 = figure.add_subplot(1, 2, 1)
        ax1.set_xlabel("N")
        ax1.set_ylabel("time")
        ax1.plot(step_ls, time_ls)

        ax2 = figure.add_subplot(1, 2, 2)
        ax2.plot(np.log(step_ls), np.log(time_ls))
        ax2.set_xlabel("log(N)")
        ax2.set_ylabel("log(time)")
        figure.tight_layout()
        figure.savefig("dft_time.png")


if __name__ == "__main__":
    test_dft(500, 2)