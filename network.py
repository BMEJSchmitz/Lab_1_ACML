import numpy as np
from layer import Layer


class Network:
    def __init__(self, layer_info, alpha, reg_lambda):
        self.layers = []
        self.layers.append(Layer(layer_info[0], layer_info[1], 'input'))
        self.layers.append(Layer(layer_info[1], layer_info[2], 'hidden'))
        self.layers.append(Layer(layer_info[2], 0, 'output'))

        self.alpha = alpha
        self.reg_lambda = reg_lambda

    def forward_propagation(self, inputs):
        for input in inputs:
            for i in range(0, len(self.layers) - 1):
                if i == 0:
                    weights = self.layers[i].thetas
                    activations = np.append(1, input)
                    z = np.matmul(np.transpose(weights), activations)
                else:
                    weights = self.layers[i].thetas
                    activations = np.append(1, activation)
                    z = np.matmul(np.transpose(weights), activations)  # add bias term to input
                activation = self.sigmoid(z)
                self.layers[i+1].set_activations(activation)

    def backward_propagation(self, inputs):
        for item in inputs:
            # last layer
            activations = self.layers[2].get_activations()
            z = np.multiply(activations, np.subtract(np.ones(activations.shape), activations))
            small_delta = np.multiply(np.subtract(activations, item), z).reshape(1, -1)

            # hidden layer
            activations = np.append(1, self.layers[1].get_activations()).reshape(-1, 1)
            v = np.matmul(activations, small_delta)
            big_delta = self.layers[1].get_errors() + v
            self.layers[1].set_errors(big_delta)
            z = np.multiply(activations, np.subtract(np.ones(activations.shape), activations))
            small_delta = np.multiply(np.matmul(self.layers[1].get_thetas(), small_delta.transpose()), z)

            # input layer
            activations = np.append(1, item)
            big_delta = np.multiply(np.transpose(activations), small_delta)
            big_delta = np.delete(big_delta, 0, 0)
            self.layers[0].set_errors(self.layers[0].get_errors() + big_delta.transpose())

        # update weights
        regularisation = np.ones((9, 3)) * (self.reg_lambda * self.layers[0].get_thetas())
        regularisation[0] = np.zeros(3)
        update = (1/len(inputs)) * (self.layers[0].get_errors() + regularisation)
        theta = self.layers[0].get_thetas() - self.alpha * update
        self.layers[0].set_thetas(theta)

        regularisation = np.ones((4, 8)) * (self.reg_lambda * self.layers[1].get_thetas())
        regularisation[0] = np.zeros(8)
        update = (1 / len(inputs)) * (self.layers[1].get_errors() + regularisation)
        theta = self.layers[1].get_thetas() - self.alpha * update
        self.layers[1].set_thetas(theta)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def train(self, inputs, epochs):
        for epoch in range(0, epochs):
            self.forward_propagation(inputs)
            self.backward_propagation(inputs)

    def test(self, input):
        activation = input
        for i in range(0, len(self.layers) - 1):
            weights = self.layers[i].thetas
            activations = np.append(1, activation)
            z = np.matmul(np.transpose(weights), activations)
            activation = self.sigmoid(z)

        return activation