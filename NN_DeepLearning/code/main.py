from code.mnist_loader import load_data_wrapper
from code.network2 import Network


def main(size,
         epochs,
         mini_batch_size,
         eta,
         lmbda,
         monitor_evaluation_cost=False,
         monitor_evaluation_accuracy=True,
         monitor_training_cost=False,
         monitor_training_accuracy=True,
         monitor_weight_vector_length=False,
         regularization='L1'):
    train, val, test = load_data_wrapper()
    net = Network(size)
    net.large_weight_initializer()
    net.SGD(training_data=train,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
            eta=eta,
            evaluation_data=test,
            lmbda=lmbda,
            monitor_evaluation_cost=monitor_evaluation_cost,
            monitor_evaluation_accuracy=monitor_evaluation_accuracy,
            monitor_training_cost=monitor_training_cost,
            monitor_training_accuracy=monitor_training_accuracy,
            monitor_weight_vector_length=monitor_weight_vector_length,
            regularization=regularization)


if __name__ == '__main__':
    size = [784, 100, 100, 10]
    epochs = 30
    mini_batch_size = 30
    eta = 4.5
    lmbda = 0.5
    monitor_evaluation_cost = False
    monitor_evaluation_accuracy = True
    monitor_training_cost = False
    monitor_training_accuracy = True
    monitor_weight_vector_length = False
    regularization = 'L2'
    main(size,
         epochs,
         mini_batch_size,
         eta,
         lmbda,
         monitor_evaluation_cost,
         monitor_evaluation_accuracy,
         monitor_training_cost,
         monitor_training_accuracy,
         monitor_weight_vector_length,
         regularization)
