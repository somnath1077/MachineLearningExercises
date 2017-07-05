from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

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
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, regression_model.predict(X_test), color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()