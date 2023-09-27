class Model:
    def __init__(self, layers: list, loss):
        self.layers = [layers]
        self.loss = loss

    def add(self, layers:list):
        self.layers.append(layers)
