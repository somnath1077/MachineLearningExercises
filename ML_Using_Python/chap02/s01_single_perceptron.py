#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:12:08 2020

@author: somnath
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron


iris = load_iris()
X = iris.data
# Create data for a binary classifier
y = (iris.target == 0).astype(np.int)  

per_clf = Perceptron()
per_clf.fit(X, y)
y_pred = per_clf.predict([[5.6, 3.0, 5.1, 1.8]])