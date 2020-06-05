import os

import keras
import numpy as np
from keras import models, layers
from keras.optimizers import adam

os.environ['TF_KERAS'] = '1'

model = models.Sequential()
model.add(layers.Dense(32, input_shape=(2,), activation='sigmoid'))
model.add(layers.Dense(32, activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))

opt = adam(lr=0.001, decay=1e-6)

X = np.zeros(shape=(10000, 3))
for i in range(X.shape[0]):
    x1 = int(np.random.randint(2, size=1))
    x2 = int(np.random.randint(2, size=1))
    y = int(x1 ^ x2)
    X[i] = [x1, x2, y]

X_train = X[:8000, :2]
y_train = X[:8000, 2]
X_test = X[8000:, :2]
y_test = X[8000:, 2]

model.compile(optimizer=opt,
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=10)
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))
