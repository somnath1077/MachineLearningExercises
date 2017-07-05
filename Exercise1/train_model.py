from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def train_model(X_train, y_train):
    regression_model = LinearRegression()
    regression_model.fit(X=X_train, y=y_train)
    return regression_model

def predict(regression_model, X_test):
    return regression_model.predict(X_test)

def prediction_accuracy(regression_model, X_test, y_test):
    prediction = regression_model.predict(X_test)
    mean = np.mean((prediction - y_test) ** 2)
    var = regression_model.score(X_test, y_test)
    return mean, var

def plot_regression(regression_model, X_test, y_test):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=X_test[:, [0]], ys=X_test[:, [1]], zs=y_test, color='black')
    ax.scatter(xs=X_test[:, [0]], ys=X_test[:, [1]], zs=regression_model.predict(X_test), color='blue')

    ax.set_xlabel('X_label')
    ax.set_ylabel('Y_label')
    ax.set_zlabel('Z_label')

    plt.xticks(())
    plt.yticks(())

    plt.show()