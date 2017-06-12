import matplotlib

from utils import inverse_hessian, gradient, logistic, create_weight_vector
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def delta_theta(training_inputs, training_targets, query_point, theta, tau, lamb):
    w_vector = create_weight_vector(training_inputs, query_point, tau)
    inv_hessian = inverse_hessian(training_inputs, query_point, theta, w_vector, lamb)
    grad = gradient(training_inputs, training_targets, query_point, theta, w_vector, lamb)
    return np.matmul(inv_hessian, grad)


def newton_raphson(training_inputs: np.ndarray,
                   training_targets: np.ndarray,
                   query_point: np.ndarray,
                   tau: np.dtype(float),
                   lamb: np.dtype(float)):
    theta = np.ones((training_inputs.shape[1], 1))
    theta_prime = np.zeros((theta.shape[0], 1))
    tolerance = 1E-6

    while min(np.absolute(theta - theta_prime)) > tolerance:
        print("Difference = ", min(np.absolute(theta - theta_prime)))
        theta_prime = theta
        theta = theta - delta_theta(training_inputs,
                                    training_targets,
                                    query_point,
                                    theta, tau, lamb)

    return theta


def prediction(training_inputs: np.ndarray,
               training_targets: np.ndarray,
               query_point: np.ndarray,
               tau: np.dtype(float),
               lamb: np.dtype(float)):
    theta = newton_raphson(training_inputs,
                           training_targets,
                           query_point,
                           tau,
                           lamb)
    prob = logistic(query_point, theta)
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


def draw(x, y, z):
    xs = np.squeeze(np.asarray(x))
    ys = np.squeeze(np.asarray(y))
    z = np.squeeze(np.asarray(z))

    colors = ['red', 'green']

    plt.figure()
    plt.scatter(xs, ys, c=z, cmap=matplotlib.colors.ListedColormap(colors))

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(xs, ys, z)
    plt.show()

if __name__ == '__main__':
    training_inputs, training_targets = load_data()
    tau = 20.0
    lamb = 1E-4
    pred = np.zeros((100, 1))
    query_points = np.zeros((100, 3))
    for idx in range(100):
        query_points[idx] = np.array([1, np.random.uniform(-1, 1), np.random.uniform(-0.5, 0.6)]).reshape((1, 3))
        pred[idx][0] = prediction(training_inputs, training_targets, query_points[idx], tau, lamb)

    draw(x=query_points[:, [1]], y=query_points[:, [2]], z=pred)
