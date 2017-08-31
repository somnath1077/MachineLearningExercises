import numpy as np


def optimize_theta(X, y, lamb):
    n_cols = X.shape[1]

    theta = np.matrix(n_cols * [1]).T
    change_in_theta = 1

    while change_in_theta > 1e-5:
        theta_new = np.matrix.copy(theta)

        for i in len(theta):
            X_i = X[:, i]
            X_term = X_i * (X * theta_new - y)

            denom = X_i.T * X_i
            numerator_neg = lamb - X_term
            numerator_pos = -1 * lamb - X_term

            theta_i_neg = min(numerator_neg / denom, 0)
            theta_i_pos = max(numerator_pos / denom, 0)

            theta_new[i] = theta_i_pos
            obj_theta_i_pos = 0.5 * np.linalg.norm(X * theta_new - y) ** 2 + lamb * sum(
                [abs(theta_i) for theta_i in theta_new])

            theta_new[i] = theta_i_neg
            obj_theta_i_neg = 0.5 * np.linalg.norm(X * theta_new - y) ** 2 + lamb * sum(
                [abs(theta_i) for theta_i in theta_new])

            if obj_theta_i_pos < obj_theta_i_neg:
                theta_new[i] = theta_i_pos
            else:
                theta_new[i] = theta_i_neg

        change_in_theta = np.matrix.norm(theta - theta_new)
        theta = np.matrix.copy(theta_new)

    return theta
