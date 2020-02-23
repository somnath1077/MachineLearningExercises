from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

def create_samples(f, left: float, right: float, num_samples: int):
    x_vals = np.linspace(left, right, num_samples)
    return [(x, f(x)) for x in x_vals]

def sigmoid(z: np.array):
    return 1.0 / (1 + np.exp(-1.0 * z))

def build_network(samples: List[Tuple[float, float]]):
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
        weight2 = 0.5 * (f_x - curr_neuron_val)

        curr_neuron_val = f_x - curr_neuron_val

        ret[idx] = [weight1, bias1, weight2]
    return ret

def evaluate_network(samples: List[Tuple[float, float]], network: np.array):
    ret = []
    for x, f_x in samples:
        net_val = 0.0
        for idx in range(network.shape[0]):
            weight1 = network[idx][0]
            bias1 = network[idx][1]
            weight2 = network[idx][2]

            net_val += sigmoid(weight1 * x + bias1) * weight2
        ret.append((x, net_val))

    return ret

def estimate_loss(samples: List[Tuple[float, float]], network_vals: List[Tuple[float, float]]):
    real_vals = np.array([f_x for _, f_x in samples])
    est_vals = np.array([f_x for _, f_x in network_vals])

    return np.mean((real_vals - est_vals)**2)


if __name__ == '__main__':

    def f(x):
        return x**2


    samples = create_samples(f, left=-4.0, right=4.0, num_samples=100)
    net = build_network(samples)
    net_vals = evaluate_network(samples, net)

    x_vals = []
    f_x =[]
    net_x = []

    for i in range(len(samples)):
        x_vals.append(samples[i][0])
        f_x.append(samples[i][1])
        net_x.append(net_vals[i][1])

    plt.plot(x_vals, f_x, color='blue')
    plt.plot(x_vals, net_x, color='red')
    plt.show()
    print(estimate_loss(samples, net_vals))