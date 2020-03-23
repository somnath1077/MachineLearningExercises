from NN_DeepLearning.code.network3 import FullyConnectedLayer, \
    SoftmaxLayer, load_data_shared, Network


def conv_main():
    train, validation, test = load_data_shared()
    mini_batch_size = 10
    epochs = 30
    net = Network([FullyConnectedLayer(n_in=784, n_out=100), SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(train,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
            eta=0.1,
            validation_data=validation,
            test_data=test,
            lmbda=0.04)


if __name__ == '__main__':
    conv_main()
