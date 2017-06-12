import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from utils import inverse_hessian, gradient, logistic, create_weights


def delta_theta(inputs, labels, query, theta, tau, lamb):
    weights = create_weights(inputs, query, tau)
    inv_hessian = inverse_hessian(inputs, query, theta, weights, lamb)
    grad = gradient(inputs, labels, query, theta, weights, lamb)
    return np.matmul(inv_hessian, grad)


def newton_raphson(inputs: np.ndarray, labels: np.ndarray, query: np.ndarray, tau: float,
                   lamb: float):
    theta = np.ones((inputs.shape[1], 1))
    theta_next = np.zeros((theta.shape[0], 1))
    tolerance = 1E-6
    learning_rate = 1E-2

    while min(np.absolute(theta - theta_next)) > tolerance:
        print("Difference = ", min(np.absolute(theta - theta_next)))
        theta_next = theta
        theta = theta - learning_rate * delta_theta(inputs, labels, query, theta, tau, lamb)
        learning_rate *= 0.1

    return theta


def prediction(inputs: np.ndarray, labels: np.ndarray, query: np.ndarray, tau: float,
               lamb: float):
    theta = newton_raphson(inputs, labels, query, tau, lamb)
    prob = logistic(query, theta)
    return int(prob > 0.5)


def load_data():
    training_inputs = load_data_to_numpy_array('data/x.dat')
    training_inputs = np.c_[np.ones(training_inputs.shape[0]), training_inputs]
    training_targets = load_data_to_numpy_array('data/y.dat')
    return training_inputs, training_targets


def load_data_to_numpy_array(filename):
    data = []
    with open(filename, 'rb') as f:
        for line in f:
            item = line.rstrip()
            data.append([float(x) for x in item.split()])
    return np.array(data)


def draw(x, y, pred):
    xs = np.squeeze(np.asarray(x))
    ys = np.squeeze(np.asarray(y))
    z = np.squeeze(np.asarray(pred))

    colors = ['red', 'green']

    plt.figure()
    plt.scatter(x=xs, y=ys, c=z, cmap=matplotlib.colors.ListedColormap(colors))

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(xs, ys, z)
    plt.show()


if __name__ == '__main__':
    inputs, labels = load_data()
    tau = 0.01
    lamb = 1E-4
    pred = np.zeros((100, 1))
    query = np.zeros((100, 3))
    for idx in range(100):
        query[idx] = np.array([1, np.random.uniform(-0.5, 0.5),
                               np.random.uniform(-0.1, 0.1)]).reshape((1, 3))
        pred[idx][0] = prediction(inputs, labels, query[idx], tau, lamb)

    draw(x=query[:, [1]], y=query[:, [2]], pred=pred)
