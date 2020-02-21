#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 07:43:00 2020

@author: somnath
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z: np.array):
    return 1 / (1 + np.exp(-1 * z))


def weighted_input(w: np.array, b: np.array, x: np.array):
    assert w.shape[1] == x.shape[0]
    assert w.shape[0] == b.shape[0]
    assert x.shape[1] == b.shape[1] == 1
    return np.dot(w, x) + b


def create_input(weight, bias, left, right, num_pts=1000):
    return np.array([weight * x + bias for x in np.linspace(left, right, num_pts)])

if __name__ == '__main__':
    # w = np.array([2.0, 3.0]).reshape((2, 1))
    # b = np.array([1.0, 1.0]).reshape((2, 1))
    # x = np.array([5.0]).reshape((1, 1))
    #
    # w_x = weighted_input(w, b, x)
    # sig = sigmoid(w_x)
    # print(sig)
    left = -0.0
    right = 1.0
    num_pts = 1000
    wt_bias = [(9.0, -4.0), (8.0, -4.0), (7.0, -4.0), (6.0, -4.0), (5.0, -4), (999.0, -400.0)]
    for weight, bias in wt_bias:
        x_vals = np.linspace(left, right, num_pts)
        y_vals = sigmoid(np.array([weight * x + bias for x in x_vals]))
        plt.plot(x_vals, y_vals, label=f'weight = {weight}, bias = {bias}')
        plt.legend()
    plt.show()
