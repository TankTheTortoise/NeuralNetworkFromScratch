import MyNeuralNet
import MyNeuralNet.layers as layers
from MyNeuralNet.models import Model
from MyNeuralNet.loss_functions import *

model = Model(
    [
        layers.Dense(2, 2, 'sigmoid'),
        layers.Dense(2, 3, 'relu'),
        layers.Dense(3, 2, 'sigmoid')
    ],
    mse
)

data = [
    [0, 0],
    [1, 1]
]

model.fit(data[0], data[1], 1, 0.1)
