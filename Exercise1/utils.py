import numpy as np


def z_vector(training_inputs: np.ndarray,
             training_targets: np.ndarray,
             theta: np.ndarray,
             query_point: np.ndarray,
             weight_vector: np.ndarray):
    assert training_inputs.shape[0] == training_targets.shape[0]
    assert weight_vector.shape[0] == training_targets.shape[0]

    z = np.zeros((training_inputs.shape[0], 1))
    for idx, input in enumerate(training_inputs):
        z[idx] = weight_vector[idx] * (training_targets[idx] - logistic(theta, input))
    return z


def gradient(training_inputs: np.ndarray,
             training_targets: np.ndarray,
             query_point: np.ndarray,
             theta: np.ndarray,
             weight_vector: np.ndarray,
             lamb: float):
    """

    :param training_inputs: a matrix, each row of which is an input vector (x)
    :param training_targets: a column vector, whose coefficients are
        the targets corresponding to the row of the training_inputs
    :param query_point: the vector at which the gradient is to be evaluated
    :param theta: the vector of parameters, relative to which the gradient
        is being computed
    :param weight_vector: the vector of weights
    :param lamb: regularization parameter
    :return: gradient at the point theta
    """
    z = z_vector(training_inputs, training_targets, theta, query_point, weight_vector)
    return np.matmul(training_inputs.transpose(), z) - lamb * theta


def hessian(training_inputs: np.ndarray,
            query_point: np.ndarray,
            theta: np.ndarray,
            weight_vector: np.ndarray,
            lamb: float):
    D = np.zeros((training_inputs.shape[0], training_inputs.shape[0]))
    I = np.identity(training_inputs.shape[1])
    for idx, row in enumerate(training_inputs):
        D[idx][idx] = -1 * weight_vector[idx] * \
                      logistic_prime(theta, row)
    return np.matmul( np.matmul(training_inputs.transpose(), D), training_inputs ) \
           - lamb * I


def inverse_hessian(training_inputs: np.ndarray,
                    query_point: np.ndarray,
                    theta: np.ndarray,
                    weight_vector: np.ndarray,
                    lamb: float):
    from numpy.linalg import inv
    return inv(hessian(training_inputs, query_point, theta, weight_vector, lamb))


def logistic(theta: np.ndarray, query_point: np.ndarray):
    theta = np.squeeze(np.asarray(theta.transpose()))
    query = np.squeeze(np.asarray(query_point))
    return 1 / (1 + np.exp(-1 * theta.dot(query)))


def logistic_prime(theta: np.ndarray, query_point: np.ndarray):
    return logistic(theta, query_point) * (1 - logistic(theta, query_point))


def create_weight_vector(training_inputs, query_point, tau):
    w_vector = np.zeros((training_inputs.shape[0], 1))
    for idx, input in enumerate(training_inputs):
        w_vector[idx] = weight(input, query_point, tau)
    return w_vector


def weight(input: np.ndarray, query: np.ndarray, tau: float):
    diff = (query - input).reshape(-1, 1)
    norm_squared = np.linalg.norm(diff)**2
    return np.math.exp(-1 * norm_squared / 2 * tau * tau)
