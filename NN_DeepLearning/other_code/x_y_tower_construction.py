from typing import List

import numpy as np

from other_code.basic_3d_plot import surface_plot


def sigmoid(z: np.array):
    return 1.0 / (1 + np.exp(-1 * z))


def xy_tower(x_int: List[float], y_int: List[float]):
    """
        Given open intervals on the x-axis x_int and the y-axis y_int,
        this function constructs a neural network that outputs = 1
        iff (x, y) \in x_int \cross y_int and 0 otherwise.
    :return:
    """
    w_x1 = w_x2 = w_y1 = w_y2 = 10000.0

    b_x1 = - x_int[0] * w_x1
    b_x2 = - x_int[1] * w_x2

    b_y1 = - y_int[0] * w_y1
    b_y2 = - y_int[1] * w_y2

    h = 10000.0

    ret = np.zeros((4, 3))
    ret[0] = [w_x1, b_x1, h]
    ret[1] = [w_x2, b_x2, -h]
    ret[2] = [w_y1, b_y1, h]
    ret[3] = [w_y2, b_y2, -h]
    return ret


def evaluate_net(z: List[float], net: np.array):
    hidden_layer_output_x = sigmoid(net[:2, 0] * z[0] + net[:2, 1]) * net[:2, 2]
    hidden_layer_output_y = sigmoid(net[2:, 0] * z[1] + net[2:, 1]) * net[2:, 2]
    # print(sigmoid(net[:2, 0] * z[0] + net[:2, 1]))
    # print(sigmoid(net[2:, 0] * z[0] + net[2:, 1]))
    bias_of_output_node = - 3 * net[0, 2] / 2.0

    wt_inp = np.sum(hidden_layer_output_x) + np.sum(hidden_layer_output_y)
    output = sigmoid(wt_inp + bias_of_output_node)
    return output


if __name__ == '__main__':
    x_interval = [1.0, 5.0]
    y_interval = [-2.0, 3.0]
    net = xy_tower(x_interval, y_interval)
    x_vals = np.arange(-3.0, 6.0, 0.1).reshape((-1, 1))
    y_vals = np.arange(-3.0, 6.0, 0.1).reshape((-1, 1))
    # z_vals = np.array([evaluate_net([x, y], net) for x in x_vals for y in y_vals]).reshape((-1, 1))
    # x_vals = [1, 2, 3, 9]
    # y_vals = [4, 5, 6]
    X, Y = np.meshgrid(x_vals, y_vals)
    z_vals = np.array([evaluate_net([x, y], net) for x in np.nditer(X) for y in np.nditer(Y)]).reshape(X.shape)
    # print(x_vals)
    # print(X)
    # print(Y)
    surface_plot(X, Y, Z)
    # xy_list = [[2.0, 0.0, 1.0], [1.01, -1.99, 1.0], [0.0, 0.0, 0.0], [-2.0, 1.0, 0.0]]
    # for x, y, answer in xy_list:
    #     net_eval = evaluate_net([x, y], net)
    #     if net_eval != answer:
    #         print(f'Failed {x}, {y}, {answer}. Got {net_eval}')
    #     else:
    #         print(f'Passed {x}, {y}, {answer}')
