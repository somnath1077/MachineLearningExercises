import itertools
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Embedding, Flatten, Dense, SimpleRNN
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


def tokenize_and_pad(text_list: List[str], vocab_size=10000, padding='post', verbose=False):
    # take the vocab_size most commonly used words
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(text_list)
    sequences = tokenizer.texts_to_sequences(text_list)

    max_len = max([len(s) for s in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding=padding)

    if verbose:
        print('Padded sequences: ')
        print(padded_sequences)
    return tokenizer, padded_sequences


def model1(vocab_sz, text_len, embedding_len=32):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_sz, output_dim=embedding_len, input_length=text_len))
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def model2(vocab_sz, text_len, embedding_len=32):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_sz, output_dim=embedding_len, input_length=text_len))
    model.add(SimpleRNN(units=8))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(docs, labels, model=None):
    _, padded_sequences = tokenize_and_pad(docs)
    max_len = len(padded_sequences[0])
    words = set(itertools.chain(*padded_sequences))
    if not model:
        m = model1(len(words), max_len)
    else:
        m = model(len(words), max_len)
    print(m.summary())

    history = m.fit(padded_sequences, labels, epochs=100, verbose=0)
    loss, accuracy = m.evaluate(padded_sequences, labels)

    return history, loss, accuracy


def plot_accuracy(history):
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    epochs = range(1, len(train_acc) + 1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 8))

    hist_keys = history.history.keys()
    val_data = 'val_acc' in hist_keys and 'val_loss' in hist_keys
    if val_data:
        val_acc = history.history['val_acc']
        val_loss = history.history['val_loss']
        ax0.plot(epochs, val_acc, 'ro', label='Validation accuracy')
        ax1.plot(epochs, val_loss, 'r', label='Validation loss')

    ax0.plot(epochs, train_acc, 'b-', label='Training accuracy')
    ax1.plot(epochs, train_loss, 'b', label='Training loss')

    for ax in (ax0, ax1):
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy/Loss')
        ax.legend()
    plt.show()


if __name__ == '__main__':
    hist, loss, acc = train_model(docs, labels, model2)
    print(f'Loss: {loss}, Accuracy: {acc}')
    # print(hist.history)
    plot_accuracy(hist)
