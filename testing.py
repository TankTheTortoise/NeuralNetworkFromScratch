import MyNeuralNet.layers as layers
from MyNeuralNet.models import Model
from MyNeuralNet.loss_functions import *
import pickle

with open("hello.pickle", "rb") as file:
    model = pickle.load(file)
print(model.layers[0].weights)
