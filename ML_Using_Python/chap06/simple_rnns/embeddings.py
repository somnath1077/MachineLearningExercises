from typing import List

import numpy as np
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
# define documents
from keras_preprocessing.sequence import pad_sequences

docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
# define labels
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


def tokenize_and_pad(text_list: List[str], vocab_size=10000, padding='post'):
    # take the vocab_size most commonly used words
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(text_list)
    sequences = tokenizer.texts_to_sequences(text_list)
    print(sequences)

    max_len = max([len(t) for t in text_list])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding=padding)
    print(padded_sequences)


max_len = 4
model = Sequential()
model.add(Embedding(input_dim=20, output_dim=3, input_length=max_len))
model.add(Flatten())
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())

padded_sequences = tokenize_and_pad(docs)
model.fit(padded_sequences, labels, epochs=100, verbose=0)
loss, accuracy = model.evaluate(padded_sequences, labels)

print(f'Loss: {loss}, Accuracy: {accuracy}')
word_embeddings = model.layers[0].get_weights()
print(word_embeddings[0])
