from typing import List

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import imdb
from keras.layers import Embedding, Dense, SimpleRNN
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
# define documents
from keras_preprocessing.sequence import pad_sequences


def tokenize_and_pad(text_list: List[str], vocab_size=10000, padding='post', max_len=None, verbose=False):
    # take the vocab_size most commonly used words
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(text_list)
    sequences = tokenizer.texts_to_sequences(text_list)

    if not max_len:
        max_len = max([len(s) for s in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding=padding)

    if verbose:
        print('Padded sequences: ')
        print(padded_sequences)
    return tokenizer, padded_sequences


def model(vocab_sz, text_len, x_train, y_train, embedding_len=32):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_sz, output_dim=embedding_len, input_length=text_len))
    model.add(SimpleRNN(units=32, return_sequences=True))
    model.add(SimpleRNN(units=32))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
    return model, history


def plot_accuracy(history):
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    epochs = range(1, len(train_acc) + 1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 8))

    hist_keys = history.history.keys()
    val_data = 'val_accuracy' in hist_keys and 'val_loss' in hist_keys
    if val_data:
        val_acc = history.history['val_accuracy']
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


def process_imdb_data(num_words=10000, max_len=500):
    """
        num_words: the number of words to use as features
        max_len: the maximum length of each review (all reviews must be of the same length)
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    train_seq = pad_sequences(x_train, maxlen=max_len)
    test_seq = pad_sequences(x_test, maxlen=max_len)

    return (train_seq, y_train), (test_seq, y_test)


def evaluate(y_pred, y_test):
    if len(y_pred) == 0:
        return
    assert len(y_pred) == len(y_test)

    acc = np.mean(np.abs(y_pred - y_test))
    print(f'Accuracy = {acc}')


if __name__ == '__main__':
    num_words = 10000
    max_len = 500
    (x_train, y_train), (x_test, y_test) = process_imdb_data(num_words, max_len)
    mod, hist = model(vocab_sz=num_words, text_len=max_len, x_train=x_train, y_train=y_train)
    y_pred = mod.predict(x_test)
    evaluate(y_pred, y_test)
    plot_accuracy(hist)
