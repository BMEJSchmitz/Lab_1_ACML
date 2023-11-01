import numpy as np
from layer import Layer


class Network:
    def __init__(self, layers, alpha, reg_lambda):
        self.layers = []
        for i in range(1, len(layers)-1):
            self.layers[i] = Layer(layers[i], layers[i-1])
        self.alpha = alpha
        self.reg_lambda = reg_lambda

    def forward_propagation(self, activation):
        for layer in self.layers:
            z = layer.thetas * np.transpose(np.append(1, activation))  # add bias term to input
            activation = self.sigmoid(z)
            layer.set_activations(activation)

    def backward_propagation(self):
        return 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, inputs, epochs):
        for epoch in range (0, epochs):
            self.forward_propagation(inputs)
            self.backward_propagation()

        return 0

