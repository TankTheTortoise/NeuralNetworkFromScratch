import MyNeuralNet
import MyNeuralNet.layers as layers
from MyNeuralNet.models import Model
from MyNeuralNet.loss_functions import *

x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

model = Model(
    [
        layers.Dense(2, 2, 'sigmoid'),
        layers.Dense(2, 3, 'relu'),
        layers.Dense(3, 1, 'relu')
    ],
    mse
)



model.fit(x_train, y_train, 100, 0.01)
