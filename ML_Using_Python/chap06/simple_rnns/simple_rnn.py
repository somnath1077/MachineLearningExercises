import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

some_data = np.random.normal(size=(40, 32, 10))
print(some_data.shape)
model = Sequential()
model.add(SimpleRNN(32, return_sequences=True))
model(some_data)
print(model.summary())
