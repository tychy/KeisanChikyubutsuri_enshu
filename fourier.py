import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def coef(k, case=1):
    sign = (-1) ** k

    if case == 1:
        pik = np.pi * k
        return 2 / pik / pik * (sign - 1)
    else:
        return 1 / k / 1.0j / np.pi * (1 - sign)


def answer(x, case=1):
    res = []
    if case == 1:
        for item in x:
            if item <= np.pi:
                value = -1 + 2 * item / np.pi
            else:
                value = 3 - 2 * item / np.pi
            res.append(value)
        return res
    else:
        for item in x:
            if item <= np.pi:
                value = 1
            else:
                value = -1
            res.append(value)
        return res


def plot_fourier(case=1, kmax=100):
    sns.set_theme()
    n = 10000
    theta = np.linspace(0, 2 * np.pi, n + 1)
    # 0 から 2*π まで (n+1)点のデータ(n分割)の配列を作る
    z = np.zeros_like(theta)
    ans = answer(theta, case)
    error_ls = []
    step_ls = []
    figure = plt.figure()
    for k in range(1, kmax + 1):
        z = z + np.real(coef(k, case) * np.exp(1.0j * k * theta))  # 波数1と2の重ね合わせ
        z = z + np.real(coef(-k, case) * np.exp(1.0j * -k * theta))  # 波数1と2の重ね合わせ
        error_ls.append(np.mean(np.fabs(ans - z)))
        step_ls.append(k)
        if k in [1, 5, 10]:
            plt.plot(theta, np.real(z), label="N={}".format(k))

    if case == 2:
        plt.plot([0, 2 * np.pi],[1.18, 1.18], "red", linestyle='dashed',label="1.18") # normal way

    plt.plot(theta, np.real(z), label="N={}".format(kmax))
    plt.xlim(0.0, 2 * np.pi)  # 横軸の表示範囲の指定
    plt.ylim(-2.0, 2.0)  # 縦軸の表示範囲の指定
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.legend()
    plt.savefig("fourier_case{}.png".format(case))

    # error の描画
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
    figure.savefig("fourier_error_case{}.png".format(case))


if __name__ == "__main__":
    plot_fourier(case=1, kmax=1000)
    plot_fourier(case=2, kmax=1000)
