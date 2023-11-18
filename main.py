import MyNeuralNet.layers as layers
from MyNeuralNet.models import Model
from MyNeuralNet.loss_functions import *
from MyNeuralNet.data_utils import *
import pickle
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv("data/mnist_train.csv")
train = np.array(train).astype('float32')
train = np.array(train)
train /= 255
x_train, y_train = train[:, 1:], train[:, 0]
y_train = one_hot(y_train)
x_train=x_train.reshape(-1, 1, 28*28)

test = pd.read_csv("data/mnist_test.csv")
test.astype("int32")
test = np.array(test)
x_test, y_test = test[:, 1:], test[:, 0]
y_test = one_hot(y_test)
x_test=x_test.reshape(-1, 1, 28*28)

model = Model(
    [
        layers.Dense(28 * 28, 128, 'relu'),
        layers.Dense(128, 128, 'relu'),
        layers.Dense(128, 10, 'sigmoid')
    ],
    mse
)

"""SOLVED - To do: Find why model isn't training- 
The learning rate was too low so it ran into the vanishing gradient problem"""

model.fit(x_train, y_train, 1, 0.1)
model.export("/home/tortoise/PycharmProjects/NeuralNetwork/hello.pickle")

with open('/home/tortoise/PycharmProjects/NeuralNetwork/hello.pickle', 'rb') as file:
    hello: Model = pickle.load(file)

for i in range(9):
    if np.argmax(hello.predict(x_test[i])) == np.argmax(y_test[i]):
        print(True)
