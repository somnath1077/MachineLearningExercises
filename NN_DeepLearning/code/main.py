from code.mnist_loader import load_data_wrapper
from code.network import Network


def main(size, epochs, mini_batch_size, eta):
    train, val, test = load_data_wrapper()
    net = Network(size)
    net.SGD(training_data=train,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
            eta=eta,
            test_data=test)


if __name__ == '__main__':
    size = [784, 128, 10]
    epochs = 100
    mini_batch_size = 10
    eta = 3.0
    main(size, epochs, mini_batch_size, eta)
