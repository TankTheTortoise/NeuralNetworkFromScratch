import numpy as np


def sigmoid(input, d=False):
    sig = 1 / (1 + np.exp(-input))
    if not d:
        return sig
    if d:
        return sig * (1 - sig)


def relu(input, d=False):
    if not d:
        input[input <= 0] = 0
        return input
    if d:
        input[input > 0] = 1
        input[input < 0] = 0
        return input


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

    def backpropagation(self, output_error, learning_rate):
        self.weights = output_error*sigmoid(self.input)


hello = Dense(2, 2, "relu")
hello.forward_propagation([1, 2])

print(hello.weights, hello.output)
