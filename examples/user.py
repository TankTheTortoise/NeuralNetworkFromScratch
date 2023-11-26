from PIL import Image, ImageChops
import numpy as np
import pickle
from MyNeuralNet.data_utils import *

file_path = "../images/one.jpg"
image = Image.open(file_path).convert('L')
image = image.resize((28, 28))
image = ImageChops.invert(image)

image = np.array(image)
image = image / 255
image = image.reshape(1, -1)

with open('../models/model.pickle', 'rb') as f:
    hello = pickle.load(f)
print(np.argmax(hello.predict(image)))
