import numpy as np

"""
    Broadcasting:
    When possible, and if there’s no ambiguity, the smaller tensor will be broadcasted 
    to match the shape of the larger tensor. Broadcasting consists of two steps:
        1. Axes (called broadcast axes) are added to the smaller tensor to match the ndim of the larger tensor.
        2. The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor.
"""


# Naive implementation of broadcasting
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        x[i] += y

    return x


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


if __name__ == '__main__':
    X = np.zeros(shape=(5, 3))
    y = np.array([1, 2, 3])

    print(f"X = {X}")
    print(f"y = {y}")
    print(f"X + y = {X + y}")
    print(naive_add_matrix_and_vector(X, y))
