import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def gen_data(theta):
    arr = np.sin(theta) + np.sin(np.sqrt(2) * theta)
    return arr


def power_spectral(t, n, f_n):
    f_k = []
    for idx in range(n):
        print("idx", idx)
        curf = 0
        for i in range(n):
            curf += f_n[i] * np.exp(-1.0j * 2 * np.pi * idx * i / n)
        curf = curf * t / n
        f_k.append(curf)
    f_k = np.array(f_k)
    print(f_k)
    f_k = f_k.real ** 2 + f_k.imag ** 2
    return f_k / t


def plot_spectral(t, n):
    sns.set_theme()
    theta = np.linspace(0, 2 * t, n + 1)
    data = gen_data(theta)
    power = power_spectral(t, n, data)
    omega = [2 * np.pi * k / t for k in range(n)]
    plt.plot(omega, power)
    plt.savefig("power.png")


if __name__ == "__main__":
    plot_spectral(100, 100)