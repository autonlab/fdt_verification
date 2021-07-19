"""
File: util.py
Author(s): Jack Good
Created: Tue Jul 14 14:28:23 EDT 2021
Description: Utilities for the demos and experiments.
Acknowledgements:
Copyright (c) 2021 Carnegie Mellon University
This code is subject to the license terms contained in the code repo.
"""

import numpy as np
import pickle

def load_model(dataset_name):
    with open("saved_models/%s.pkl" % (dataset_name), "rb") as f:
        return pickle.load(f)

def load_robustness_test_points(dataset_name):
    yx = np.genfromtxt("robustness_test_points/%s.csv" % (dataset_name), delimiter=",")
    return yx[:,1:], yx[:,0].astype(int)

def make_grid(box, n):
    x0 = np.linspace(box[0][0], box[0][1], n)
    x1 = np.linspace(box[1][0], box[1][1], n)
    x0, x1 = np.meshgrid(x0, x1)
    return np.concatenate((x0.reshape(-1,1), x1.reshape(-1,1)), axis=1)