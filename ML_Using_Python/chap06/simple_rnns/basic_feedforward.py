import numpy as np

time_steps = 5
input_features = 2
output_features = 1

inputs = np.random.random((time_steps, input_features))
state_t = np.zeros((output_features, ))

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features, ))

successive_outputs = []

for input_t in inputs:
    output_t = np.dot(W, input_t) + np.dot(U, state_t) + b
    state_t = output_t

    successive_outputs.append(output_t)

final_output = np.concatenate(successive_outputs, axis=0)
print(final_output)