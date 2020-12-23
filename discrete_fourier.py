import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def gen_data(n):
    arr = np.random.rand(n)
    return arr


def coef(f_ls):
    size = len(f_ls)
    coef_ls = []
    for i in range(size):
        c_k = 0
        for idx, item in enumerate(f_ls):
            c_k += item * np.exp(-1.0j * 2 * np.pi * i * idx / size)
        coef_ls.append(c_k / size)
    return coef_ls


def dft(coef_ls):
    size = len(coef_ls)
    f_ls = []
    for i in range(size):
        f_n = 0
        for idx, item in enumerate(coef_ls):
            f_n += coef_ls[idx] * np.exp(1.0j * 2 * np.pi * i * idx / size)
        f_ls.append(np.real(f_n / size))
    return f_ls


def test_dft():
    arr = gen_data(10)
    print("input", arr)
    coef_ls = coef(arr)
    f_ls = dft(coef_ls)
    print("out", f_ls)


if __name__ == "__main__":
    test_dft()