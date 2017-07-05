from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MODEL_TYPE = {'linear': LinearRegression,
              'logistic': LogisticRegression}

def train_model(X_train, y_train, **kwargs):
    _model = LinearRegression
    if kwargs:
        _model = MODEL_TYPE[kwargs['model_type']]

    model = _model()
    model.fit(X=X_train, y=y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)


def prediction_accuracy(model, X_test, y_test):
    prediction = model.predict(X_test)
    mean = np.mean((prediction - y_test) ** 2)
    var = model.score(X_test, y_test)
    return mean, var


def plot_model_predictions(model, X_test, y_test):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=X_test[:, [0]], ys=X_test[:, [1]], zs=y_test, color='black')
    ax.scatter(xs=X_test[:, [0]], ys=X_test[:, [1]], zs=model.predict(X_test), color='blue')

    ax.set_xlabel('X_label')
    ax.set_ylabel('Y_label')
    ax.set_zlabel('Z_label')

    plt.xticks(())
    plt.yticks(())

    plt.show()
