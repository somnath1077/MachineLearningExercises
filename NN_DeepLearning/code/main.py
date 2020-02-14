from code.mnist_loader import load_data_wrapper
from code.network2 import Network


def main(size, epochs, mini_batch_size, eta, lmbda):
    train, val, test = load_data_wrapper()
    net = Network(size)
    net.large_weight_initializer()
    net.SGD(training_data=train,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
            eta=eta,
            evaluation_data=test,
            lmbda=lmbda,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=False,
            monitor_training_accuracy=True,
            monitor_weight_vector_length=False,
            regularization='L1')


if __name__ == '__main__':
    size = [784, 100, 100, 10]
    epochs = 30
    mini_batch_size = 30
    eta = 3.0
    lmbda = 2.0
    main(size, epochs, mini_batch_size, eta, lmbda)
