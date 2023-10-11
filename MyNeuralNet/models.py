import pickle
import numpy as np


class Model:
    def __init__(self, layers: list, loss):
        self.layers = layers
        self.loss = loss

    def add(self, layers: list):
        self.layers.append(layers)

    def predict(self, inputs: np.array):
        samples = len(inputs)
        output = inputs[0]
        for layer in self.layers:
            output = layer.forward_propagation(output)
        results = np.array(output)

        for i in range(1, samples):
            output = inputs[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            results = np.append(results, output, axis=0)
        return results

    def accuracy(self, x_test, y_test):
        guess = np.argmax(self.predict(x_test), axis=-1)
        return np.mean(np.equal(guess, np.argmax(y_test, axis=-1)))
        # add one to array for every matching argmax in x_test and y_test

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Find loss for display only
                err += self.loss(y_train[j], output, d=False)

                error = self.loss(y_train[j], output, d=True)
                for layer in reversed(self.layers):
                    error = layer.back_propagation(error, learning_rate)

            # Calculate average error on all samples
            err /= samples
            print(f'epoch {i + 1}/{epochs}   error={err * 100}%')

    def export(self, path):
        with open(f'{path}', 'wb') as file:
            pickle.dump(self, file)
