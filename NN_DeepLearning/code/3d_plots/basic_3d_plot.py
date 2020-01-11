#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:31:30 2020

@author: somnath
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def scatter_plot(X, Y, Z):
    _ = plt.figure()
    ax = plt.axes(projection="3d")
    
    ax.scatter3D(X, Y, Z, cmap='hsv')
    
    plt.show()
    
def wireframe_plot(X, Y, Z, color='green'):
    _ = plt.figure()
    ax = plt.axes(projection="3d")
    
    ax.plot_wireframe(X, Y, Z, color=color)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
    
def surface_plot(X, Y, Z):
    _ = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                    cmap='winter', edgecolor='none')
    ax.set_title('surface')
    
    plt.show()

def plot_function(f, 
                  x_lo=-10.0, x_hi=10.0, x_pts=50, 
                  y_lo=-10.0, y_hi=10.0, y_pts=50):
    x_vals = np.linspace(x_lo, x_hi, x_pts)
    y_vals = np.linspace(y_lo, y_hi, y_pts)
    
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)
    
    scatter_plot(X, Y, Z)
    wireframe_plot(X, Y, Z)
    surface_plot(X, Y, Z)

def f(x, y):
    return 2 * x**2 + 3 * y**2

def f2(x, y):
    return np.sin(np.sqrt(x**2 + y**2))


plot_function(f)
plot_function(f2)