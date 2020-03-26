from NN_DeepLearning.code.network3 import FullyConnectedLayer, ConvPoolLayer, \
    SoftmaxLayer, load_data_shared, Network
from theano.tensor import tanh
from theano.tensor.nnet.nnet import relu


def conv_main():
    train, validation, test = load_data_shared()
    mini_batch_size = 10
    epochs = 60
    net = Network([ConvPoolLayer(input_shape=(mini_batch_size, 1, 28, 28),
                                 filter_shape=(20, 1, 5, 5),
                                 poolsize=(2, 2),
                                 activation_fn=relu),
                   ConvPoolLayer(input_shape=(mini_batch_size, 20, 12, 12),
                                 filter_shape=(40, 20, 5, 5),
                                 poolsize=(2, 2),
                                 activation_fn=relu),
                   FullyConnectedLayer(n_in=40 * 4 * 4,
                                       n_out=100,
                                       activation_fn=relu),
                   SoftmaxLayer(n_in=100,
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
