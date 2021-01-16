import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def sum_square(vector):
    return vector[0] ** 2 + vector[1] ** 2


def duv(position):
    denominator = sum_square(position) ** 1.5
    return np.array([-position[0] / denominator, -position[1] / denominator])


def dxy(velocity):
    return velocity


def runge_kutta_one(position, velocity, dh):
    k_one_pos = dh * dxy(velocity)
    k_one_vel = dh * duv(position)
    position += k_one_pos
    velocity += k_one_vel
    return position, velocity


def runge_kutta_four(position, velocity, dh):
    k_one_pos = dh * dxy(velocity)
    k_one_vel = dh * duv(position)
    k_two_pos = dh * dxy(velocity + k_one_vel / 2)
    k_two_vel = dh * duv(position + k_one_pos / 2)
    k_three_pos = dh * dxy(velocity + k_two_vel / 2)
    k_three_vel = dh * duv(position + k_two_pos / 2)
    k_four_pos = dh * dxy(velocity + k_three_vel)
    k_four_vel = dh * duv(position + k_three_pos)
    position += k_one_pos / 6 + k_two_pos / 3 + k_three_pos / 3 + k_four_pos / 6
    velocity += k_one_vel / 6 + k_two_vel / 3 + k_three_vel / 3 + k_four_vel / 6
    return position, velocity


def execute(dh, t_end):
    t = 0
    position = np.array([3.0, 0.0])
    velocity = np.array([0.3, 0.2])
    energy_ls = []
    t_ls = []
    x_ls = []
    y_ls = []
    u_ls = []
    v_ls = []

    energy_init = sum_square(velocity) / 2 - 1.0 / (sum_square(position) ** 0.5)
    while t <= t_end:
        energy = sum_square(velocity) / 2 - 1.0 / (sum_square(position) ** 0.5)
        energy_ls.append(np.fabs(energy - energy_init))
        t_ls.append(t)

        position, velocity = runge_kutta_four(position, velocity, dh)
        x_ls.append(position[0])
        y_ls.append(position[1])
        u_ls.append(velocity[0])
        v_ls.append(velocity[1])

        t += dh
    x_ls = np.asarray(x_ls)
    y_ls = np.asarray(y_ls)
    u_ls = np.asarray(u_ls)
    v_ls = np.asarray(v_ls)
    t_ls = np.array(t_ls)
    energy_ls = np.array(energy_ls)
    return x_ls, y_ls, u_ls, v_ls, t_ls, energy_ls


def plot_pos():
    dh = 0.01
    t_end = 1000
    x_ls, y_ls, u_ls, v_ls, t_ls, energy_ls = execute(dh, t_end)

    sns.set_theme()

    fig = plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x_ls, y_ls, label="dh={}".format(dh))
    plt.savefig("kepler.png")


def plot_energy():
    t_end = 1000
    dh_ls = [0.001, 0.005, 0.01, 0.05, 0.1]
    sns.set_theme()
    fig = plt.figure()

    for dh in dh_ls:
        x_ls, y_ls, u_ls, v_ls, t_ls, energy_ls = execute(dh, t_end)
        plt.plot(np.log10(t_ls[1:]), np.log10(energy_ls[1:]), label="dh={}".format(dh))
    plt.xlabel("log10(t)")
    plt.ylabel("log10(|E- E_0|)")
    plt.legend()
    plt.savefig("energy.png")


if __name__ == "__main__":
    plot_pos()
    plot_energy()