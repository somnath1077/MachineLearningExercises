from sklearn import svm
import numpy as np

# Code based on: https://martin-thoma.com/svm-with-sklearn/
# This code was copied from the said site for the purpose of learning ONLY!

def main():
    mnist_data = get_data()
    X, y = scale_data(mnist_data)
    data_dict = split_data_train_test(X, y)

    clf = svm.SVC(probability=False, kernel='rbf', C=2.8, gamma=0.0073)
    examples = len(data_dict['train']['X'])

    print('fitting data ... ')
    clf.fit(data_dict['train']['X'][:examples], data_dict['train']['y'][:examples])
    print('finished fitting data ...')

    analyse(clf, data_dict)


def analyse(clf, data_dict):
    from sklearn import metrics

    predicted = clf.predict(data_dict['test']['X'])

    print("Confusion matrix: \n %s" %
          metrics.confusion_matrix(data_dict['test']['y'], predicted))
    print("Accuracy: %0.4f" % metrics.accuracy_score(data_dict['test']['y'], predicted))

    try_id = 1
    out = clf.predict(data_dict['test']['X'][try_id])  # clf.predict_proba
    print("out: %s" % out)

    size = int(len(data_dict['test']['X'][try_id]) ** (0.5))

    view_image(data_dict['test']['X'][try_id].reshape((size, size)),
               data_dict['test']['y'][try_id])

def view_image(image, label=''):
    from matplotlib.pyplot import show, imshow, cm
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()


def get_data():
    from sklearn.datasets import fetch_mldata

    print('fetching data ...')
    mnist = fetch_mldata('MNIST original')
    print('finished fetching data ...')

    return mnist

def scale_data(mnist):
    X = mnist.data
    y = mnist.target

    return (X / 255.0) * 2 - 1, y

def split_data_train_test(X, y):
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split

    X, y = shuffle(X, y, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return {'train': {'X': X_train,
                      'y': y_train},
            'test':  {'X': X_test,
                      'y': y_test}}


if __name__ == '__main__':
    main()