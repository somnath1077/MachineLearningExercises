#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:57:11 2021

@author: This is code for an example on the Newton-Raphson technique
for a function from R^2 to R^2. Taken from Advanced Calculus (Page 349). 
"""
import numpy as np


def f(z):
    x = z[0][0]
    y = z[1][0]
    
    x_1 = - 13 + x - 2 * y + 5 * y**2 - y**3
    y_1 = - 29 + x - 14 * y + y**2 + y**3

    return np.array([x_1, y_1])


def numerator_expr_1(y):
    return 3 * y**2 + 2 * y - 14

def numerator_expr_2(y):
    return 3 * y**2 - 10 * y + 2

def denom_expr(y):
    return 6 * y**2 - 8 * y - 12

def f_jacobian(z):
    y = z[1][0]
    a = numerator_expr_1(y) / denom_expr(y)
    b = numerator_expr_2(y) / denom_expr(y)
    c = -1 / denom_expr(y)
    d = 1 / denom_expr(y)
    
    return np.array([[a, b], [c, d]])


def dist(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))

def newton(x0, y0, tol=1e-4):
    z_old = np.array([x0, y0]).reshape(2, 1)
    z_new = np.array([2 * x0, 2 * y0]).reshape(2, 1)
    
    d = dist(z_old, z_new)
    iter = 0
    
    while d > tol:
        z_new = z_old - np.matmul(f_jacobian(z_old), f(z_old).reshape(2, 1))
        d = dist(z_new, z_old)
        print(f'iteration: {iter:3d}: x: {z_old[0][0]:10.5f}, y: {z_old[1][0]:10.5f}')
        z_old = z_new
        iter += 1
    
    return z_new

if __name__ == '__main__':
    newton(10, 8)
    
    
    

