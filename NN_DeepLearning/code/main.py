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
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True,
            monitor_weight_vector_length=True)


if __name__ == '__main__':
    size = [784, 100, 10]
    epochs = 50
    mini_batch_size = 30
    eta = 3.0
    lmbda = 0.00
    main(size, epochs, mini_batch_size, eta, lmbda)
