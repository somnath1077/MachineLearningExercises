import numpy as np
import os

dir = os.path.dirname(__file__)
data_dir = dir + '/data/'
training_inputs_file =  os.path.join(data_dir, 'x.dat')
training_targets_file = os.path.join(data_dir, 'y.dat')

def load_data():
    training_inputs = load_data_to_numpy_array(training_inputs_file)
    inputs = np.c_[np.ones(training_inputs.shape[0]), training_inputs]
    targets = load_data_to_numpy_array(training_targets_file)
    return inputs, targets


def load_data_to_numpy_array(filename):
    data = []
    with open(filename, 'rb') as f:
        for line in f:
            item = line.rstrip()
            data.append([float(x) for x in item.split()])
    return np.array(data)
