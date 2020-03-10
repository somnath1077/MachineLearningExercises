"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
reg, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

# Libraries
# Standard library
import json
import random
import sys
# Third-party libraries
from typing import List, Tuple

import numpy as np


def create_dropout_weights(weight_matrix_list: List[np.array], dropout: float):
    ret = []
    for w in weight_matrix_list:
        nrow = w.shape[0]
        ncol = w.shape[1]
        sz = nrow * ncol
        n = int(np.rint(dropout * sz))

        B = np.ones((sz, 1), dtype=np.float64)
        idx = np.random.choice(sz, size=n, replace=False)
        B[idx] = 0.0

        B = B.reshape((nrow, ncol))
        ret.append(w * B)
    return ret


# Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * tanh_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``x`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return a - y


# Main Network class
class Network(object):

    def __init__(self, sizes, activation='sigmoid'):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = list()
        self.weights = list()
        self.default_weight_initializer()
        self.cost = CrossEntropyCost
        self.transform_y = False
        if activation == 'tanh':
            self.cost = QuadraticCost
            self.transform_y = True

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            # a = sigmoid(np.dot(w, a) + b)
            a = tanh(np.dot(w, a) + b)
        return a

    def SGD(self,
            training_data,
            epochs,
            mini_batch_size,
            eta,
            decay=0.001,
            lmbda=0.0,
            dropout=0.1,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            monitor_weight_vector_length=False,
            regularization='L2'):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        reg parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        training_data = list(training_data)
        evaluation_data = list(evaluation_data)

        if evaluation_data:
            n_data = len(evaluation_data)

        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,
                                       eta,
                                       lmbda,
                                       n=len(training_data),
                                       regularization=regularization,
                                       dropout=dropout)
            print("Epoch %s training complete" % (j + 1))

            cost = self.total_cost(evaluation_data, lmbda, convert=True)
            evaluation_cost.append(cost)
            if len(evaluation_cost) >= 2 and evaluation_cost[-1] > evaluation_cost[-2]:
                eta = eta * (1 - decay)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))

            if monitor_evaluation_cost:
                # cost = self.total_cost(evaluation_data, lmbda, convert=True)
                # evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))

            if monitor_weight_vector_length:
                wt_mat_sq = [np.square(w) for w in self.weights]
                wt_mat_sum = [np.sum(wt_mat) for wt_mat in wt_mat_sq]
                print(f"Length of weight vector = {np.sqrt(np.sum(wt_mat_sum))}")

        return (evaluation_cost,
                evaluation_accuracy,
                training_cost,
                training_accuracy)

    def update_mini_batch(self,
                          mini_batch: List[Tuple[np.array, np.array]],
                          eta: float,
                          lmbda: float,
                          n: int,
                          regularization: str = 'L2',
                          dropout: float = 0.1):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the reg parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        X = []
        Y = []
        for x, y in mini_batch:
            X.append(x)
            Y.append(y)
        X = np.column_stack(X)
        Y = np.column_stack(Y)
        delta_nabla_b, delta_nabla_w = self.backprop_full_matrix_dropout(X, Y, dropout)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # for x, y in mini_batch:
        #    delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        #    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        #    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        if regularization == 'L2':
            self.weights = [(1 - eta * (lmbda / n)) * w -
                            (eta / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]
        elif regularization == 'L1':
            self.weights = [w - eta * (lmbda / n) * np.sign(w) -
                            (eta / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]
        else:
            print('Please specify proper reg!')
            sys.exit(1)

        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop_full_matrix(self, X, Y):
        """
           This full matrix version returns a tuple ``(nabla_b, nabla_w)``
           representing the mean gradient for the cost function C computed
           over all the examples in the mini-batch represented by the
           arrays X and Y.

           X: a numpy array whose columns ar the examples of the mini-batch
              That is, X = [x_1, x_2, ..., x_m]
           Y: a numpy array whose columns represent the labels
              y_1, y_2, ..., y_m corresponding to the examples x_1, ..., x_m

           ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of
           numpy arrays, similar to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = X
        # list to store all the activations, layer by layer
        activations = [X]
        zs = []  # list to store all the x vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            # The ith column of x is the weighted input corresponding
            # to the ith example in the mini-batch
            # assert x.shape[1] == X.shape[1]
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass

        # The ith element of activations is a np array of dimension:
        # number of neurons in ith layer * mini-batch size
        # assert len(activations) == len(self.sizes)
        delta = self.cost.delta(zs[-1], activations[-1], Y)
        nabla_b[-1] = np.sum(delta, axis=1).reshape(nabla_b[-1].shape)

        A = activations[-2].transpose()
        nabla_w[-1] = np.dot(delta, A)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1).reshape(nabla_b[-l].shape)

            act = activations[-l - 1].transpose()
            nabla_w[-l] = np.dot(delta, act)

        return nabla_b, nabla_w

    def backprop_full_matrix_dropout(self, X, Y, dropout: float = 0.1):
        """
           This full matrix version returns a tuple ``(nabla_b, nabla_w)``
           representing the mean gradient for the cost function C computed
           over all the examples in the mini-batch represented by the
           arrays X and Y.

           X: a numpy array whose columns ar the examples of the mini-batch
              That is, X = [x_1, x_2, ..., x_m]
           Y: a numpy array whose columns represent the labels
              y_1, y_2, ..., y_m corresponding to the examples x_1, ..., x_m

           ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of
           numpy arrays, similar to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = X
        # list to store all the activations, layer by layer
        activations = [X]
        zs = []  # list to store all the x vectors, layer by layer

        biases = create_dropout_weights(self.biases, dropout)
        weights = create_dropout_weights(self.weights, dropout)
        for b, w in zip(biases, weights):
            z = np.dot(w, activation) + b
            # The ith column of x is the weighted input corresponding
            # to the ith example in the mini-batch
            # assert x.shape[1] == X.shape[1]
            zs.append(z)
            # activation = sigmoid(x)
            activation = tanh(z)
            activations.append(activation)

        # backward pass

        # The ith element of activations is a np array of dimension:
        # number of neurons in ith layer * mini-batch size
        # assert len(activations) == len(self.sizes)
        delta = self.cost.delta(zs[-1], activations[-1], Y)
        nabla_b[-1] = np.sum(delta, axis=1).reshape(nabla_b[-1].shape)

        A = activations[-2].transpose()
        nabla_w[-1] = np.dot(delta, A)
        for l in range(2, self.num_layers):
            z = zs[-l]
            # sp = sigmoid_prime(x)
            sp = tanh_prime(z)

            delta = np.dot(weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1).reshape(nabla_b[-l].shape)

            act = activations[-l - 1].transpose()
            nabla_w[-l] = np.dot(delta, act)

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
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the x vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
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

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        transform_y = True if we need a +1 -1 vector instead of a 0-1 vector
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y, self.transform_y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


# Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


# Miscellaneous functions
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


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1 - np.tanh(z) ** 2
