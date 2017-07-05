from data_loader import load_data
from train_model import train_model, prediction_accuracy, plot_regression

X, y = load_data()
assert X.shape[0] == y.shape[0]

# separate into training set and test set
num_rows = X.shape[0]
last_training_row = int((2 * num_rows) / 3)
X_train = X[:last_training_row]
y_train = y[:last_training_row]
assert X_train.shape[0] == y_train.shape[0]

X_test = X[last_training_row:]
y_test = y[last_training_row:]
assert X_test.shape[0] == y_test.shape[0]

model = train_model(X_train, y_train)
mean, var = prediction_accuracy(model, X_test, y_test)
print("MSE = {}; Var = {}". format(mean, var))
plot_regression(model, X_test, y_test)