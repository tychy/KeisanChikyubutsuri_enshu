import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_center(u, grid):
    k = np.zeros_like(u)
    for i in range(grid - 1):
        k[i] = (u[i + 1] - u[i - 1]) / 2  # b.c.は-1で自動的に入る
    k[grid - 1] = (u[0] - u[grid - 2]) / 2
    return k


def get_upwind(u, grid):
    k = np.zeros_like(u)
    for i in range(grid - 1):
        k[i] = u[i] - u[i - 1]  # b.c.は-1で自動的に入る
    k[grid - 1] = u[grid - 1] - u[grid - 2]
    return k


def get_downwind(u, grid):
    k = np.zeros_like(u)
    for i in range(grid - 1):
        k[i] = u[i + 1] - u[i]  # b.c.は-1で自動的に入る
    k[grid - 1] = u[0] - u[grid - 1]
    return k


def center(dt):

    sns.set_theme()
    grid = 100
    x = np.linspace(0, 1, grid, endpoint=True)
    u = np.power(np.sin(x * np.pi), 100)
    c = 1
    t = 0
    t_end = 1
    figure = plt.figure()
    plt.plot(x, u, label="t=0")

    while t < t_end:
        u_prev = np.copy(u)
        k_one = dt * grid * c * get_center(u_prev, grid)
        u = u_prev - k_one
        t += dt
    figure = plt.figure()
    plt.plot(x, u, label="t={}".format(t_end))
    plt.savefig("adv_center.png")


def center_rk(dt):

    sns.set_theme()
    grid = 100
    x = np.linspace(0, 1, grid, endpoint=True)
    u = np.power(np.sin(x * np.pi), 100)
    c = 1
    t = 0
    t_end = 1
    figure = plt.figure()
    plt.plot(x, u, label="t=0")

    while t < t_end:
        u_prev = np.copy(u)
        k_one = -dt * grid * c * get_center(u_prev, grid)
        k_two = -dt * grid * c * get_center(u_prev + k_one / 2, grid)
        k_three = -dt * grid * c * get_center(u_prev + k_two / 2, grid)
        k_four = -dt * grid * c * get_center(u_prev + k_three, grid)
        u = u_prev + (k_one + 2 * k_two + 2 * k_three + k_four) / 6
        t += dt
    plt.plot(x, u, label="t={}".format(t_end))
    plt.legend()
    plt.savefig("adv_center_rk.png")


def upwind_rk(dt):

    sns.set_theme()
    grid = 100
    x = np.linspace(0, 1, grid, endpoint=True)
    u = np.power(np.sin(x * np.pi), 100)
    c = 1
    t = 0
    t_end = 1

    figure = plt.figure()
    plt.plot(x, u, label="t=0")
    while t < t_end:
        u_prev = np.copy(u)
        k_one = -dt * grid * c * get_upwind(u_prev, grid)
        k_two = -dt * grid * c * get_upwind(u_prev + k_one / 2, grid)
        k_three = -dt * grid * c * get_upwind(u_prev + k_two / 2, grid)
        k_four = -dt * grid * c * get_upwind(u_prev + k_three, grid)
        u = u_prev + (k_one + 2 * k_two + 2 * k_three + k_four) / 6
        t += dt
    plt.plot(x, u, label="t={}".format(t_end))
    plt.legend()
    plt.savefig("adv_upwind_rk.png")


def downwind_rk(dt):
    sns.set_theme()

    grid = 100
    x = np.linspace(0, 1, grid, endpoint=True)
    u = np.power(np.sin(x * np.pi), 100)
    c = 1
    t = 0
    t_end = 1

    figure = plt.figure()
    plt.plot(x, u, label="t=0")
    while t < t_end:
        u_prev = np.copy(u)
        k_one = -dt * grid * c * get_downwind(u_prev, grid)
        k_two = -dt * grid * c * get_downwind(u_prev + k_one / 2, grid)
        k_three = -dt * grid * c * get_downwind(u_prev + k_two / 2, grid)
        k_four = -dt * grid * c * get_downwind(u_prev + k_three, grid)
        u = u_prev + (k_one + 2 * k_two + 2 * k_three + k_four) / 6
        t += dt
    plt.plot(x, u, label="t={}".format(t_end))
    plt.legend()
    plt.savefig("adv_downwind_rk.png")


def main():
    dt = 0.01
    center(dt)
    center_rk(dt)
    upwind_rk(dt)
    downwind_rk(dt)


if __name__ == "__main__":
    main()