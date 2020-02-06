#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 08:59:23 2020

@author: somnath
"""
import numpy as np
import matplotlib.pyplot as plt


def softmax(z: np.array):
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z)
    return e_z / sum_e_z


def plot_activations(z_initial: np.array,
                     idx: int,
                     min_val: float,
                     max_val: float,
                     num_pts: int = 1000):
    z_idx_range = np.linspace(min_val, max_val, num_pts)
    num_inputs = len(z_initial)

    activations_list = np.zeros((num_pts, num_inputs))
    for i in range(num_pts):
        x = z_idx_range[i]
        z_initial[idx] = x
        f_z = softmax(z_initial)

        for j in range(num_inputs):
            activations_list[i] = f_z

    for col in range(activations_list.shape[1]):
        plt.plot(z_idx_range,
                 activations_list[:, col],
                 label=f'Activation {col}')
    plt.xlabel(f'Range of values for z_{idx}')
    plt.ylabel(f'Neuron Activations')
    plt.legend()


if __name__ == '__main__':
    idx = 3
    z_initial = np.array([-1.0, 0.0, 1.0, -4.0])
    plot_activations(z_initial, idx=3, min_val=-4.0, max_val=4.0)
