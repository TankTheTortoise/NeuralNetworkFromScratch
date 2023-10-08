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
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Find loss for display only
                err += self.loss(y_train[j], output)

                error = self.loss(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.back_propagation(error, learning_rate)

            # Calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
