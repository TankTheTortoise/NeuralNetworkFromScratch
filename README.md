# NeuralNetworkFromScratch
A feed forward neural network with backpropagation. 

## Example
### Importing the library

1. Clone the MyNeuralNet folder into your project directory.
2. Import it into your program using the import statements below.

```
import MyNeuralNet.layers as layers
from MyNeuralNet.models import Model
from MyNeuralNet.loss_functions import *
from MyNeuralNet.creator import load
```

### Creating the Model
The model using a Keras like creation
```
model = Model(
  [
    Layers.Dense(input_shape, output_shape, activation_function),
    Layers.Dense(input_shape, output_shape, activation_function),
    Layers.Dense(input_shape, output_shape, activation_function)
  ],
  loss_function
)
```
### Importing a Model
```
model = load(path)
```

### Training the Model
```
model.fit(x_train, y_train, epochs, learning_rate)
```

### Testing the Model
```
print(model.predict(test_sample))
```

### Accuracy of the Model
```
print(model.accuracy(x_test, y_test))
```

### Export Model
```
model.export(path)
```
