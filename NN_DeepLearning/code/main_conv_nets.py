from NN_DeepLearning.code.network3 import FullyConnectedLayer, ConvPoolLayer, \
    SoftmaxLayer, load_data_shared, Network


def conv_main():
    train, validation, test = load_data_shared()
    mini_batch_size = 10
    epochs = 30
    net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                                 filter_shape=(20, 1, 5, 5),
                                 poolsize=(2, 2)),
                   ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                                 filter_shape=(40, 20, 5, 5),
                                 poolsize=(2, 2)),
                   FullyConnectedLayer(n_in=40 * 4 * 4, n_out=100),
                   SoftmaxLayer(n_in=100, n_out=10)],
                  mini_batch_size)
    net.SGD(train,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
            eta=0.1,
            validation_data=validation,
            test_data=test,
            lmbda=0.01)


if __name__ == '__main__':
    conv_main()
