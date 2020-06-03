import numpy as np

"""
    Broadcasting:
    When possible, and if there’s no ambiguity, the smaller tensor will be broadcasted 
    to match the shape of the larger tensor. Broadcasting consists of two steps:
        1. Axes (called broadcast axes) are added to the smaller tensor to match the ndim of the larger tensor.
        2. The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor.
"""

X = np.zeros(shape=(5, 3))
y = np.array([1, 2, 3])

print(X.ndim)
print(y.ndim)
