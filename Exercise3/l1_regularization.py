import numpy as np

from typing import TypeVar

matrix = TypeVar('np.matrixlib.defmatrix.matrix')

def optimize_theta(X: matrix, y: matrix, L1_param: float):
    """
    Implements the co-ordinate descent algorithm to find 
    the optimum weight vector theta that minimizes the 
    sum of squares error function + L1 reg. 
    Lambda is the parameter of the L1 reg term.
    """
    n_cols = X.shape[1]

    theta = np.matrix(np.random.rand(1, n_cols)).T
    theta_new = np.matrix(n_cols * [0]).T

    while np.linalg.norm(theta - theta_new) > 1e-5:
        theta_new = theta

        print(theta_new)

        for i in range(len(theta)):
            X_i = X[:, i]
            X_term = X_i.T * (X * theta_new - y)

            denom = X_i.T * X_i
            numerator_neg = L1_param - X_term
            numerator_pos = -1 * L1_param - X_term

            theta_i_neg = min(numerator_neg / denom, 0)
            theta_i_pos = max(numerator_pos / denom, 0)

            theta_new[i] = theta_i_pos
            obj_theta_i_pos = 0.5 * np.linalg.norm(X * theta_new - y) ** 2 + L1_param * sum(
                [abs(theta_i) for theta_i in theta_new])

            theta_new[i] = theta_i_neg
            obj_theta_i_neg = 0.5 * np.linalg.norm(X * theta_new - y) ** 2 + L1_param * sum(
                [abs(theta_i) for theta_i in theta_new])

            if obj_theta_i_pos < obj_theta_i_neg:
                theta_new[i] = theta_i_pos
            else:
                theta_new[i] = theta_i_neg

        theta = theta_new

    return theta

if __name__ == '__main__':
    X = np.matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    true_theta = np.matrix([1.0, 2.0, 1.0]).T
    y = X * true_theta + np.matrix(np.random.rand(1, 2)).T
    lamb = 0.000001
    theta = optimize_theta(X, y, lamb)
    print(theta)