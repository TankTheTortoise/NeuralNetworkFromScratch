import MyNeuralNet.layers as layers
from MyNeuralNet.models import Model
from MyNeuralNet.loss_functions import *
import pickle
from MyNeuralNet import creator

model: Model = creator.load("model.pickle")
print(model.layers[0].weights)
