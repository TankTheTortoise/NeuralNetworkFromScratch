import MyNeuralNet
import MyNeuralNet.layers as layers
from MyNeuralNet.models import Model
from MyNeuralNet.loss_functions import *

x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

model = Model(
    [
        layers.Dense(2, 2, 'relu'),
        layers.Dense(2, 3, 'relu'),
        layers.Dense(3, 1, 'sigmoid')
    ],
    mse
)

"""SOLVED - To do: Find why model isn't training- 
The learning rate was too low so it ran into the vanishing gradient problem"""
print(model.layers[2].weights)
model.fit(x_train, y_train, 100, 1)
print(model.layers[2].weights)
