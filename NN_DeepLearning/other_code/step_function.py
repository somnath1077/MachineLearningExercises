from typing import List

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z: np.array):
    return 1.0 / (1 + np.exp(-1 * z))


def x_step(x_int: List[float]):
    """
        Given open intervals on the x-axis x_int
        this function constructs a neural network that outputs = 1
        iff x \in x_int and 0 otherwise.
    :return:
    """
    w_x1 = w_x2 = 10000.0

    b_x1 = - x_int[0] * w_x1
    b_x2 = - x_int[1] * w_x2

    h = 10000.0

    ret = np.zeros((2, 3))
    ret[0] = [w_x1, b_x1, h]
    ret[1] = [w_x2, b_x2, -h]
    return ret


def evaluate_net(x: float, net: np.array):
    upper_x = sigmoid(net[0, 0] * x + net[0, 1]) * net[0, 2]
    lower_x = sigmoid(net[1, 0] * x + net[1, 1]) * net[1, 2]
    bias_of_switch_node = 0

    wt_inp = upper_x + lower_x
    output_of_switch = sigmoid(wt_inp + bias_of_switch_node)
    return output_of_switch


if __name__ == '__main__':
    x_interval = [-1.0, 4.0]
    net = x_step(x_interval)

    x_vals = np.arange(-2.0, 5.0, 0.01)
    y_vals = [evaluate_net(x, net) for x in x_vals]

    plt.plot(x_vals, y_vals, linewidth=2)
    plt.ylabel('Output of network')
    plt.xlabel(r'$x$')
    plt.show()
