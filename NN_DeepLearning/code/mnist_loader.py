"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""
import os
import gzip
# Libraries
# Standard library
import pickle

# Third-party libraries
import numpy as np


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    path_to_this = os.path.abspath('.')
    # This module is called code and so the path till the parent of this file is path_to_this[:-4]
    path_to_data = path_to_this[:-4] + "data/mnist.pkl.gz"
    with gzip.open(path_to_data, 'rb') as ff:
        u = pickle._Unpickler(ff)
        u.encoding = 'latin1'
        training_data, validation_data, test_data = u.load()

    return training_data, validation_data, test_data


def load_data_wrapper(transform_y=False):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y, transform_y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    # training_data = [(tr_inp, tr_res) for tr_inp in training_inputs
    #                 for tr_res in training_results]
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    # validation_data = [(val_inp, val_res) for val_inp in validation_inputs
    #                  for val_res in va_d[1]]
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    # test_data = [(test_inp, test_res) for test_inp in test_inputs
    #             for test_res in te_d[1]]
    return training_data, validation_data, test_data


def vectorized_result(j, transform_y=False):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    repeat_digit = 0
    if transform_y:
        repeat_digit = -1
    # e = np.zeros((10, 1))
    e = np.repeat(repeat_digit, 10).reshape((10, 1))
    e[j] = 1.0
    return e
