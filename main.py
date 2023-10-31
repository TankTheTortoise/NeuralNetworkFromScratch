

# TODO adjust data structure
"""
32x32 bitmaps are divided into nonoverlapping blocks of 4x4
]and the number of on pixels are counted in each block.
This generates an input matrix of 8x8 where each element
is an integer in the range 0..16. This reduces dimensionality
and gives invariance to small distortions.
"""




import MyNeuralNet.layers as layers
from MyNeuralNet.models import Model
from MyNeuralNet.loss_functions import *
from MyNeuralNet.creator import load
from MyNeuralNet.data_utils import *
import pickle
import matplotlib
import pandas as pd
from ucimlrepo import fetch_ucirepo
 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)


# fetch dataset
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)
print(optical_recognition_of_handwritten_digits.data.targets)
# data (as pandas dataframes)
X = np.array(optical_recognition_of_handwritten_digits.data.features)
y = np.array(optical_recognition_of_handwritten_digits.data.targets)

np.random.shuffle(X)
np.random.shuffle(y)

train_size = 0.8
# load MNIST from server
(x_train, x_test), (y_train, y_test) = split(X, train_size), split(y, train_size)

# Display values
d_x_train = x_train
d_y_train = x_train
d_x_test = x_test
d_y_test = x_test

# training data : 60000 samples
# reshape and normalize input data
print(len(x_test[0]))
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = one_hot(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = one_hot(y_test)

model = Model(
    [
        layers.Dense(28 * 28, 128, 'relu'),
        layers.Dense(128, 128, 'relu'),
        layers.Dense(128, 10, 'sigmoid')
    ],
    mse
)

if input("Type 'Create' to create new model: ") == "Create":
    """SOLVED - To do: Find why model isn't training- 
    The learning rate was too low so it ran into the vanishing gradient problem"""

    model.fit(x_train, y_train, 2, 0.01)
    model.export("/home/tortoise/PycharmProjects/NeuralNetwork/model.pickle")

    print(model.accuracy(x_test, y_test))
    # for i in range(9):
    #     print(np.argmax(hello.predict(x_test[i])))
    #     print(np.argmax(y_test[i]))
    #     plt.subplot(330 + 1 + i)
    #     plt.imshow(d_x_test[i], cmap=plt.get_cmap('gray'))
    # plt.show()
else:
    oldModel: Model = load("/home/tortoise/PycharmProjects/NeuralNetwork/hello.pickle")
    print(y_test)
    print(oldModel.accuracy(x_test, y_test))
