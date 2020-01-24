"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# Libraries
# Standard library
import random
# Third-party libraries
import sys

import numpy as np


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self,
            training_data,
            epochs,
            mini_batch_size,
            eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.

        Since both the training and the test data are zipped objects,
        in order to find out their lengths, we need to first
        convert them into a list. Python 2.7 to Python 3.7 problem!
        """
        training_data = list(training_data)
        test_data = list(test_data)

        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # X = []
        # Y = []
        # cnt = 1
        # for x, y in mini_batch:
            # print(f'{cnt}: {x.shape}')
            # print(f'{cnt}: {y.shape}')
            # cnt += 1
            # X.append(x)
            # Y.append(y)

        # X = np.array(X).reshape((X[0].shape[0], len(X)))
        # print(X.shape)
        # Y = np.array(Y).reshape((Y[0].shape[0], len(Y)))
        # print(Y.shape)
        # delta_nabla_b, delta_nabla_w = self.backprop_full_matrix_version(X, Y)
        # nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        # nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop_full_matrix_version(self, X, Y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
           mean gradient for the cost function C computed over all
           the examples in the mini-batch represented by the arrays X and Y.

           X: a numpy array whose columns ar the examples of the mini-batch
                That is, X = [x_1, x_2, ..., x_m]
           Y: a numpy array whose columns represent the labels y_1, y_2, ..., y_m
                corresponding to the examples x_1, ..., x_m

           ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of
           numpy arrays, similar to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = X
        # list to store all the activations, layer by layer
        activations = [X]
        zs = []  # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            assert z.shape[0] == w.shape[0]
            # The ith column of z is the weighted input corresponding
            # to the ith example in the mini-batch
            assert z.shape[1] == X.shape[1]
            zs.append(z)
            activation = sigmoid(z)
            assert activation.shape[0] == w.shape[0]
            assert activation.shape[1] == X.shape[1]
            activations.append(activation)

        # backward pass

        # The ith element of activations is a np array of dimension:
        # number of neurons in ith layer * mini-batch size
        assert len(activations) == len(self.sizes)
        assert zs[-1].shape[0] == self.sizes[-1]
        assert zs[-1].shape[1] == X.shape[1]
        assert activations[-1].shape == Y.shape == zs[-1].shape
        delta = self.cost_derivative(activations[-1], Y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = np.sum(delta, axis=1).reshape(nabla_b[-1].shape)

        A = activations[-2].transpose()
        for delta_x, A_x in zip(delta.T, A):
            delta_x = delta_x.reshape((delta.shape[0], 1))
            A_x = A_x.reshape((A.shape[1], 1)).T
            nabla_w[-1] += np.dot(delta_x, A_x)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1).reshape(nabla_b[-l].shape)

            act = activations[-l - 1].transpose()
            for delta_x, act_x in zip(delta.T, act):
                delta_x = delta_x.reshape((delta.shape[0], 1))
                act_x = act_x.reshape((act.shape[1], 1)).T
                nabla_w[-l] += np.dot(delta_x, act_x)

        return nabla_b, nabla_w

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        # list to store all the activations, layer by layer
        activations = [x]
        zs = []  # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y


# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
