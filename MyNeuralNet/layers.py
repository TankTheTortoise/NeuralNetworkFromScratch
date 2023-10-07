import numpy as np
from .activation_functions import *
from .loss_functions import *


class Dense:
    def __init__(self, input_nodes, output_nodes, activation_function):
        self.weights = np.random.uniform(-1, 1, [input_nodes, output_nodes])
        self.biases = np.random.uniform(-1, 1, [output_nodes])

        self.activation_funcs = {"sigmoid": sigmoid, "relu": relu}
        self.func = self.activation_funcs[activation_function]

        self.output = None
        self.input = None

    def add_activation(self, func_name, func):
        self.activation_funcs[func_name] = func

    def forward_propagation(self, input):
        self.input = input
        self.output = self.func(np.dot(input, self.weights) + self.biases)
        return self.output

    def back_propagation(self, output_error, learning_rate):
        output_error *= sigmoid(self.input)
        # The derivative of the weighted sum WX+B is equal to the input X
        input_error = np.dot(output_error, self.weights)

        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * self.biases

        return input_error
