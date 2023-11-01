import numpy as np
from layer import Layer


class Network:
    def __init__(self, layers, alpha, reg_lambda):
        self.layers = []
        for i in range(1, len(layers)-1):
            self.layers[i] = Layer(layers[i], layers[i-1])
        self.alpha = alpha
        self.reg_lambda = reg_lambda

    def forward_propagation(self, input):
        self.layers[1].thetas
        z =theta*inputs
        activation = self.sigmoid(z)
        return 0

    def backward_propagation(self):
        return 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, inputs):

        return 0

