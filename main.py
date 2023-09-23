import MyNeralNet
import MyNeralNet.layers as layers

hello = layers.Dense(2, 2, "relu")
hello.forward_propagation([1, 2])

print(hello.weights, hello.output)
