import numpy as np
import matplotlib.pyplot as plt
from sor import execute as sor
import seaborn as sns


def is_bc(i, j, kGrid):
    return i == 0 or i == kGrid or j == 0 or j == kGrid


def get_index(i, j, kGrid):
    return j - 1 + (i - 1) * (kGrid - 1)


def main():
    kGrid = 20
    matrix_size = (kGrid - 1) ** 2
    dh = 1.0 / kGrid
    u = np.zeros((kGrid + 1, kGrid + 1))
    u[:, kGrid] = 1
    for x in range(kGrid + 1):
        u[x, 0] = x * dh
    for y in range(kGrid + 1):
        u[0, y] = y * dh
        u[kGrid, y] = 1 - y * dh

    # main loop
    a = np.zeros((matrix_size, matrix_size))
    b = np.zeros((matrix_size))

    for x in range(1, kGrid):
        for y in range(1, kGrid):
            cur_index = get_index(x, y, kGrid)
            if x == 1 or x == kGrid - 1:
                for i, coef in zip([-1, 0, 1], [12, -24, 12]):
                    if is_bc(x + i, y, kGrid):
                        b[cur_index] += -1 * coef * u[x + i, y]
                    else:
                        a[cur_index, get_index(x + i, y, kGrid)] += coef
            else:
                for i, coef in zip([-2, -1, 0, 1, 2], [-1, 16, -30, 16, -1]):
                    if is_bc(x + i, y, kGrid):
                        b[cur_index] += -1 * coef * u[x + i, y]
                    else:
                        a[cur_index, get_index(x + i, y, kGrid)] += coef

            if y == 1 or y == kGrid - 1:
                for i, coef in zip([-1, 0, 1], [12, -24, 12]):
                    if is_bc(x, y + i, kGrid):
                        b[cur_index] += -1 * coef * u[x, y + i]
                    else:
                        a[cur_index, get_index(x, y + i, kGrid)] += coef
            else:
                for i, coef in zip([-2, -1, 0, 1, 2], [-1, 16, -30, 16, -1]):
                    if is_bc(x, y + i, kGrid):
                        b[cur_index] += -1 * coef * u[x, y + i]
                    else:
                        a[cur_index, get_index(x, y + i, kGrid)] += coef
    fig = plt.figure()
    sns.heatmap(np.rot90(u))
    plt.savefig("init.png")
    fig = plt.figure()
    x = sor(a, b, 1.2)
    x = np.reshape(x, (kGrid - 1, kGrid - 1))
    u[1:kGrid, 1:kGrid] = x
    u = np.rot90(u)

    sns.heatmap(u)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    # plt.imshow(u, interpolation="none")
    plt.savefig("pde.png")


if __name__ == "__main__":
    main()