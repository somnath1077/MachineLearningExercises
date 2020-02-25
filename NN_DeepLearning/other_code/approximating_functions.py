from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def create_samples(f, left: float, right: float, num_samples: int) -> List[Tuple[float, float]]:
    """
        This function takes in a function f, an interval [left, right]
        and a parameter num_samples for the number of times one wishes
        to sample from the function f in the given interval. Samples
        are taken at equidistant points in the stated interval.

        In the network to be constructed, the number of neurons in the
        the hidden layer equals the number of samples.
    """
    x_vals = np.linspace(left, right, num_samples)
    return [(x, f(x)) for x in x_vals]


def sigmoid(z: np.array) -> np.array:
    return 1.0 / (1 + np.exp(-1.0 * z))


def build_network(samples: List[Tuple[float, float]]) -> np.array:
    """
        From samples, construct a matrix whose rows look like:

            weight1, bias1, weight2

        where weight1 and bias1 are the weight and bias of the neuron
        corresponding to that row and weight2 is the weight of the link
        from that neuron to the output node.

    :return: np.array of shape (len(samples), 3)
    """
    num_samples = len(samples)
    ret = np.zeros((num_samples, 3), dtype=np.float64)

    curr_neuron_val = 0.0
    for s, idx in zip(samples, range(num_samples)):
        x = s[0]
        f_x = s[1]

        weight1 = 1000
        bias1 = - x * weight1
        weight2 = f_x - curr_neuron_val

        curr_neuron_val += weight2

        ret[idx] = [weight1, bias1, weight2]
    return ret


def evaluate_network(samples: List[Tuple[float, float]],
                     network: np.array) -> List[Tuple[float, float]]:
    ret = []
    for x, f_x in samples:
        net_val = float(np.sum(sigmoid((network[:, 0] * x + network[:, 1])) * network[:, 2]))
        ret.append((x, net_val))
    return ret


def estimate_loss(samples: List[Tuple[float, float]],
                  network_vals: List[Tuple[float, float]]) -> np.array:
    real_vals = np.array([f_x for _, f_x in samples])
    est_vals = np.array([f_x for _, f_x in network_vals])

    return np.mean((real_vals - est_vals) ** 2)


def plot_network(samples: List[Tuple[float, float]],
                 network_vals: List[Tuple[float, float]]):
    x_vals = []
    f_x = []
    net_x = []

    for i in range(len(samples)):
        x_vals.append(samples[i][0])
        f_x.append(samples[i][1])
        net_x.append(network_vals[i][1])

    plt.plot(x_vals, f_x, color='green', label='actual')
    plt.plot(x_vals, net_x, color='blue', label='network')
    plt.legend()
    plt.show()


# This is the function from Nielsen's book
def f1_nielsen(x):
    return 0.2 + 0.4 * x ** 2 + 0.3 * x * np.sin(15 * x) + 0.05 * np.cos(50 * x)


def f2(x):
    return x ** 2


if __name__ == '__main__':
    samples_for_net = create_samples(f1_nielsen, left=-5.0, right=5.0, num_samples=500)
    net = build_network(samples_for_net)
    samples_for_plot = create_samples(f1_nielsen, left=-5.0, right=5.0, num_samples=1000)
    net_vals = evaluate_network(samples_for_plot, net)

    plot_network(samples_for_plot, net_vals)
    print(estimate_loss(samples_for_plot, net_vals))
