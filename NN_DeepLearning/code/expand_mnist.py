"""expand_mnist.py
~~~~~~~~~~~~~~~~~~

Take the 50,000 MNIST training images, and create an expanded set of
250,000 images, by displacing each training image up, down, left and
right, by one pixel.  Save the resulting file to
../data/mnist_expanded.pkl.gz.

Note that this program is memory intensive, and may not run on small
systems.

"""

from __future__ import print_function

import gzip
import os.path
# Standard library
import pickle
import random

# Third-party libraries
import numpy as np

#### Libraries

print("Expanding the MNIST training set")

path_to_this = os.path.abspath('.')
path_to_data = path_to_this[:-4] + "data/mnist.pkl.gz"
path_to_expanded_data = path_to_this[:-4] + "data/mnist_expanded.pkl.gz"

print(path_to_this)
print(path_to_data)
print(path_to_expanded_data)

if os.path.exists(path_to_expanded_data):
    print("The expanded training set already exists.  Exiting.")
else:
    with gzip.open(path_to_data, 'rb') as ff:
        u = pickle._Unpickler(ff)
        u.encoding = 'latin1'
        training_data, validation_data, test_data = u.load()
    # f = gzip.open("../data/mnist.pkl.gz", 'rb')
    # training_data, validation_data, test_data = pickle.load(f)
    # f.close()
    expanded_training_pairs = []
    j = 0  # counter
    for x, y in zip(training_data[0], training_data[1]):
        expanded_training_pairs.append((x, y))
        image = np.reshape(x, (-1, 28))
        j += 1
        if j % 1000 == 0: print("Expanding image number", j)
        # iterate over data telling us the details of how to
        # do the displacement
        for d, axis, index_position, index in [
            (1, 0, "first", 0),
            (-1, 0, "first", 27),
            (1, 1, "last", 0),
            (-1, 1, "last", 27)]:
            new_img = np.roll(image, d, axis)
            if index_position == "first":
                new_img[index, :] = np.zeros(28)
            else:
                new_img[:, index] = np.zeros(28)
            expanded_training_pairs.append((np.reshape(new_img, 784), y))
    random.shuffle(expanded_training_pairs)
    expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
    print("Saving expanded data. This may take a few minutes.")
    f = gzip.open(path_to_expanded_data, "w")
    pickle.dump((expanded_training_data, validation_data, test_data), f)
    f.close()

