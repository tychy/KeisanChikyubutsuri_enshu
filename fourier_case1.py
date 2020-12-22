import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def coef(k, case=1):
    sign = (-1) ** k
    pik = np.pi * k
    return 2 / pik / pik * (sign - 1)


def answer(x, case=1):
    res = []
    for item in x:
        if item <= np.pi:
            value = -1 + 2 * item / np.pi
        else:
            value = 3 - 2 * item / np.pi
        res.append(value)
    return res


def plot_fourier(case=1, kmax=10000):
    n = 360
    theta = np.linspace(0, 2 * np.pi, n + 1)
    # 0 から 2*π まで (n+1)点のデータ(n分割)の配列を作る
    z = np.zeros_like(theta)
    ans = answer(theta, case)
    error_ls = []
    for k in range(1, kmax + 1):
        z = z + np.real(coef(k) * np.exp(1.0j * k * theta))  # 波数1と2の重ね合わせ
        z = z + np.real(coef(-k) * np.exp(1.0j * -k * theta))  # 波数1と2の重ね合わせ
        error_ls.append(np.fabs(ans - z))

    plt.plot(theta, np.real(z))  # zの実部をグラフ表示
    plt.xlim(0.0, 2 * np.pi)  # 横軸の表示範囲の指定
    plt.ylim(-2.0, 2.0)  # 縦軸の表示範囲の指定
    plt.savefig("fourier_case{}.png".format(case))
    i = 1
    while i < kmax:
        plt.plot(error_ls[i])

        i *= 2

    plt.savefig("fourier_error_case{}.png".format(case))


if __name__ == "__main__":
    plot_fourier(case=1)