import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def f(x, t):
    res = x + (x ** 2 + t ** 2) ** 0.5
    res = res / t
    return res


def runge_kutta_one(dh, t_end):
    t = 1
    x = 0
    while t <= t_end:
        x = x + dh * f(x, t)
        t += dh
    return x


def runge_kutta_two(dh, t_end):
    t = 1
    x = 0
    while t <= t_end:
        k_one = dh * f(x, t)
        # Heun法を採用
        k_two = dh * f(x + k_one / 2, t + dh / 2)
        x = x + k_one / 2 + k_two / 2
        t += dh
    return x


def runge_kutta_four(dh, t_end):
    t = 1
    x = 0
    while t <= t_end:
        k_one = dh * f(x, t)
        k_two = dh * f(x + k_one / 2, t + dh / 2)
        k_three = dh * f(x + k_two / 2, t + dh / 2)
        k_four = dh * f(x + k_three, t + dh)
        x = x + k_one / 6 + k_two / 3 + k_three / 3 + k_four / 6
        t += dh
    return x


def plot_runge_kutta():

    sns.set_theme()
    tend = 11
    one_ls = []
    two_ls = []
    four_ls = []
    ans = 60
    dh_ls = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
    for dh in dh_ls:
        one_ls.append(np.fabs(ans - runge_kutta_one(dh, tend)))
        two_ls.append(np.fabs(ans - runge_kutta_two(dh, tend)))
        four_ls.append(np.fabs(ans - runge_kutta_four(dh, tend)))
    log_dh = np.log10(dh_ls)
    plt.plot(log_dh, one_ls, label="RK1")
    plt.plot(log_dh, two_ls, label="RK2")
    plt.plot(log_dh, four_ls, label="RK4")
    plt.xlabel("np.log10(h)")
    plt.ylabel("|60 - x|")
    plt.legend()
    plt.savefig("runge_kutta_error.png")


if __name__ == "__main__":
    plot_runge_kutta()