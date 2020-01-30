#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 07:25:29 2020

@author: somnath
"""

import numpy as np
import matplotlib.pyplot as plt


def cross_entropy(y, a):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))

x = np.linspace(0.01, 0.99, 1000)
f_x = np.array([cross_entropy(i, i) for i in x])

plt.plot(x, f_x)
