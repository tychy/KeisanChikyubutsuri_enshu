import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def f(x, t):
    res = x + np.sqrt(x ** 2 + t ** 2)
    res = res / t
    return res


def runge_kutta_one(x, dh, t, t_end):
    while t < t_end:
        x = x + dh * f(x, t)
        t += dh
    return x


def runge_kutta_two(x, dh, t, t_end):
    while t < t_end:
        k_one = dh * f(x, t)
        # Heun法を採用
        k_two = dh * f(x + k_one / 2, t + dh / 2)
        x = x + k_one / 2 + k_two / 2
        t += dh
    return x


def runge_kutta_four(x, dh, t, t_end):
    while t < t_end:
        k_one = dh * f(x, t)
        k_two = dh * f(x + k_one / 2, t + dh / 2)
        k_three = dh * f(x + k_two / 2, t + dh / 2)
        k_four = dh * f(x + k_three, t + dh)
        x = x + k_one / 6 + k_two / 3 + k_three / 3 + k_four / 6
        t += dh
    return x


def runge_kutta_four_efficient(x, dh, t, t_end):
    while t < t_end:
        x_cur = x
        k = dh * f(x, t)
        x += k / 6
        k = dh * f(x_cur + k / 2, t + dh / 2)
        x += k / 3
        k = dh * f(x_cur + k / 2, t + dh / 2)
        x += k / 3
        k = dh * f(x_cur + k, t + dh)
        x += k / 6
        t += dh
    return x


def plot_runge_kutta():
    sns.set_theme()
    x, t = 0, 1
    tend = 11
    one_ls = []
    two_ls = []
    four_ls = []
    ans = 60
    dh_ls = [
        0.0001,
        0.0003,
        0.0007,
        0.001,
        0.003,
        0.007,
        0.01,
        0.03,
        0.07,
        0.1,
        0.2,
        0.5,
        1,
    ]
    for dh in dh_ls:
        one_ls.append(np.fabs(ans - runge_kutta_one(x, dh, t, tend)))
        two_ls.append(np.fabs(ans - runge_kutta_two(x, dh, t, tend)))
        four_ls.append(np.fabs(ans - runge_kutta_four_efficient(x, dh, t, tend)))
    log_dh = np.log10(dh_ls)
    plt.plot(log_dh, np.log10(one_ls), label="RK1")
    plt.plot(log_dh, np.log10(two_ls), label="RK2")
    plt.plot(log_dh, np.log10(four_ls), label="RK4")
    plt.xlabel("log10(h)")
    plt.ylabel("log10(|60 - x|)")
    plt.legend()
    plt.savefig("runge_kutta_error.png")


if __name__ == "__main__":
    plot_runge_kutta()