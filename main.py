import MyNeuralNet.layers as layers
from MyNeuralNet.models import Model
from MyNeuralNet.loss_functions import *
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from keras.datasets import mnist
from keras.utils import to_categorical

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Display values
d_x_train = x_train
d_y_train = x_train
d_x_test = x_test
d_y_test = x_test

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

model = Model(
    [
        layers.Dense(28*28, 128, 'relu'),
        layers.Dense(128, 128, 'relu'),
        layers.Dense(128, 10, 'sigmoid')
    ],
    mse
)


if input() == "Create":
    """SOLVED - To do: Find why model isn't training- 
    The learning rate was too low so it ran into the vanishing gradient problem"""

    model.fit(x_train, y_train, 1, 0.1)
    model.export("/home/tortoise/PycharmProjects/NeuralNetwork/hello.pickle")

    with open('/home/tortoise/PycharmProjects/NeuralNetwork/hello.pickle', 'rb') as file:
        hello:Model = pickle.load(file)



    for i in range(9):
        print(np.argmax(hello.predict(x_test[i])))
        print(np.argmax(y_test[i]))
        plt.subplot(330 + 1 + i)
        plt.imshow(d_x_test[i], cmap=plt.get_cmap('gray'))
        plt.show()
else:
