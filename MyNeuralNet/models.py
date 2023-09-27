class Model:
    def __init__(self, layers: list, loss):
        self.layers = layers
        self.loss = loss

    def add(self, layers: list):
        self.layers.append(layers)

    def predict(self, inputs: list):
        samples = len(inputs)
        results = []

        for i in range(samples):
            output = inputs[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            results.append(output)
        return results

    def fit(self, x_train, y_train, epochs, learning_rate):
        for i in range(epochs):
            err = 0
