from theano.tensor.nnet.nnet import relu

from NN_DeepLearning.code.network3 import FullyConnectedLayer, ConvPoolLayer, \
    SoftmaxLayer, load_data_shared, Network


def conv_main():
    train, validation, test = load_data_shared('data/mnist_expanded.pkl.gz')
    mini_batch_size = 10
    epochs = 30
    net = Network([ConvPoolLayer(input_shape=(mini_batch_size, 1, 28, 28),
                                 filter_shape=(20, 1, 5, 5),
                                 poolsize=(2, 2),
                                 activation_fn=relu),
                   ConvPoolLayer(input_shape=(mini_batch_size, 20, 12, 12),
                                 filter_shape=(40, 20, 5, 5),
                                 poolsize=(2, 2),
                                 activation_fn=relu),
                   FullyConnectedLayer(n_in=40 * 4 * 4,
                                       n_out=1000,
                                       activation_fn=relu,
                                       p_dropout=0.5),
                   FullyConnectedLayer(n_in=1000,
                                       n_out=1000,
                                       activation_fn=relu,
                                       p_dropout=0.5),
                   SoftmaxLayer(n_in=1000,
                                n_out=10)],
                  mini_batch_size)
    net.SGD(train,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
            eta=0.03,
            validation_data=validation,
            test_data=test,
            lmbda=0.1)


if __name__ == '__main__':
    conv_main()
