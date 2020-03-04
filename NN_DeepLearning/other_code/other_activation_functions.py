import matplotlib.pyplot as plt
import numpy as np


def tanh_prime(z: np.array):
    return 1 - np.tanh(z) ** 2


if __name__ == '__main__':
    x = np.linspace(-5.0, 5.0, 1000)
    # y = tanh_prime(x)
    y = np.tanh(x)
    plt.plot(x, y)
    plt.xlabel(r'$x$')
    plt.ylabel(r"$tanh(x)$")
    plt.title(r'tanh')
    plt.show()
