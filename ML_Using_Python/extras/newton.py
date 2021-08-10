#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 09:56:07 2021

@author: somnath
"""
from collections.abc import Callable


def f(x):
    return x**3 - 20


def f_prime(x):
    return 3 * x**2 

def netwon(guess: float, 
           f: Callable, 
           f_prime: Callable, 
           tol: float = 1e-4):
    x_old = guess
    x_new = 2 * guess
    
    gap = abs(x_old - x_new)
    
    while gap > tol:
        x_new = x_old - f(x_old) / f_prime(x_old)
        print(f'old: {x_old}; new: {x_new}')
        gap = abs(x_old - x_new)
        x_old = x_new
    
    return x_new

if __name__ == '__main__':
    print(netwon(2.5, f, f_prime, 1e-7))
        