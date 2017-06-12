import numpy as np


def z_vector(inputs: np.ndarray,
             labels: np.ndarray,
             theta: np.ndarray,
             query: np.ndarray,
             weights: np.ndarray):
    assert inputs.shape[0] == labels.shape[0]
    assert weights.shape[0] == labels.shape[0]

    z = np.zeros((inputs.shape[0], 1))
    for idx, input in enumerate(inputs):
        z[idx] = weights[idx] * (labels[idx] - logistic(theta, input))
    return z


def gradient(inputs: np.ndarray,
             labels: np.ndarray,
             query: np.ndarray,
             theta: np.ndarray,
             weights: np.ndarray,
             lamb: float):
    """

    :param inputs: a matrix, each row of which is an input vector (x)
    :param labels: a column vector, whose coefficients are
        the targets corresponding to the row of the training_inputs
    :param query: the query vector at which the gradient is to be evaluated
    :param theta: the vector of parameters, relative to which the gradient
        is being computed
    :param weights: the vector of weights
    :param lamb: regularization parameter
    :return: gradient at the point theta
    """
    z = z_vector(inputs, labels, theta, query, weights)
    return np.matmul(inputs.transpose(), z) - lamb * theta


def hessian(inputs: np.ndarray,
            query: np.ndarray,
            theta: np.ndarray,
            weights: np.ndarray,
            lamb: float):
    D = np.zeros((inputs.shape[0], inputs.shape[0]))
    I = np.identity(inputs.shape[1])
    for idx, row in enumerate(inputs):
        D[idx][idx] = -1 * weights[idx] * \
                      logistic_prime(theta, row)
    return np.matmul(np.matmul(inputs.transpose(), D), inputs) \
           - lamb * I


def inverse_hessian(inputs: np.ndarray,
                    query: np.ndarray,
                    theta: np.ndarray,
                    weights: np.ndarray,
                    lamb: float):
    from numpy.linalg import inv
    return inv(hessian(inputs, query, theta, weights, lamb))


def logistic(theta: np.ndarray, query: np.ndarray):
    return 1 / (1 + np.exp(-1 * np.matmul(theta.transpose(), query)))


def logistic_prime(theta: np.ndarray, query: np.ndarray):
    return logistic(theta, query) * (1 - logistic(theta, query))


def create_weights(inputs, query, tau):
    w_vector = np.zeros((inputs.shape[0], 1))
    for idx, input in enumerate(inputs):
        w_vector[idx] = weight(input, query, tau)
    return w_vector


def weight(input: np.ndarray, query: np.ndarray, tau: float):
    diff = (query - input).reshape(-1, 1)
    norm_squared = np.linalg.norm(diff)**2
    return np.math.exp(-1 * norm_squared / 2 * tau * tau)
