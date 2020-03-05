from NN_DeepLearning.code.mnist_loader import load_data_wrapper
from NN_DeepLearning.code.network2 import Network, QuadraticCost


def main(size,
         epochs,
         mini_batch_size,
         eta,
         decay,
         lmbda,
         dropout,
         monitor_evaluation_cost=False,
         monitor_evaluation_accuracy=True,
         monitor_training_cost=False,
         monitor_training_accuracy=True,
         monitor_weight_vector_length=False,
         regularization='L1'):
    train_data, val_data, test_data = load_data_wrapper(transform_y=True)
    # we are using the tanh activation function, so we need to use +1-1 vectors
    net = Network(size, activation='tanh')
    # net.large_weight_initializer()
    net.SGD(training_data=train_data,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
            eta=eta,
            decay=decay,
            evaluation_data=val_data,
            lmbda=lmbda,
            dropout=dropout,
            monitor_evaluation_cost=monitor_evaluation_cost,
            monitor_evaluation_accuracy=monitor_evaluation_accuracy,
            monitor_training_cost=monitor_training_cost,
            monitor_training_accuracy=monitor_training_accuracy,
            monitor_weight_vector_length=monitor_weight_vector_length,
            regularization=regularization)


if __name__ == '__main__':
    size = [784, 300, 10]
    epochs = 30
    mini_batch_sz = 10
    eta = 0.08
    decay = 0.0001
    lmbda = 0.01
    dropout = 0.02
    evaluation_cost = False
    evaluation_accuracy = True
    training_cost = False
    training_accuracy = False
    wt_vect_length = False
    reg = 'L2'
    main(size,
         epochs,
         mini_batch_sz,
         eta,
         decay,
         lmbda,
         dropout,
         evaluation_cost,
         evaluation_accuracy,
         training_cost,
         training_accuracy,
         wt_vect_length,
         reg)
