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
        for item in inputs:
            activation = item
            for i in range(0, len(self.layers) - 1):
                weights = self.layers[i].thetas
                mul = np.matmul(np.transpose(weights), activation.transpose())
                bias = self.layers[i].get_bias()
                z = mul.transpose() + bias
                activation = self.sigmoid(z)
                self.layers[i+1].set_activations(activation)

    def backward_propagation(self, inputs):
        for item in inputs:
            # last layer
            activations = self.layers[2].get_activations().reshape(1, -1)
            z = -self.sigmoid_derivative(activations)
            small_delta = np.multiply(activations - item, z).transpose()

            # hidden layer
            activations = self.layers[1].get_activations().reshape(-1, 1)
            v = np.matmul(activations, small_delta.reshape(1, -1))
            big_delta = self.layers[1].get_errors() + v
            self.layers[1].set_errors(big_delta)
            self.layers[1].set_bias_error(self.layers[1].get_bias_error() + small_delta.transpose())
            z = self.sigmoid_derivative(activations)
            small_delta = np.multiply(np.matmul(self.layers[1].get_thetas(), small_delta.reshape(-1, 1)), z).transpose()

            # input layer
            activations = item.reshape(-1, 1)
            v = np.matmul(activations, small_delta.reshape(1, -1))
            big_delta = self.layers[0].get_errors() + v
            self.layers[0].set_errors(big_delta)
            self.layers[0].set_bias_error(self.layers[0].get_bias_error() + small_delta)

        # update weights
        theta_update = self.layers[1].get_errors()
        reg = self.reg_lambda * self.layers[1].get_thetas()
        new_theta = self.layers[1].get_thetas() - self.alpha * (1/len(inputs) * theta_update + reg)
        self.layers[1].set_thetas(new_theta)
        self.layers[1].set_errors(np.zeros(self.layers[1].get_errors().shape))

        bias_update = self.layers[1].get_bias_error()
        new_bias = self.layers[1].get_bias() - self.alpha * (1/len(inputs) * bias_update)
        self.layers[1].set_bias(new_bias)
        self.layers[1].set_bias_error(np.zeros(self.layers[1].get_bias_error().shape))

        theta_update = self.layers[0].get_errors()
        reg = self.reg_lambda * self.layers[0].get_thetas()
        new_theta = self.layers[0].get_thetas() - self.alpha * (1 / len(inputs) * theta_update + reg)
        self.layers[0].set_thetas(new_theta)
        self.layers[0].set_errors(np.zeros(self.layers[0].get_errors().shape))

        bias_update = self.layers[0].get_bias_error()
        new_bias = self.layers[0].get_bias() - self.alpha * (1 / len(inputs) * bias_update)
        self.layers[0].set_bias(new_bias)
        self.layers[0].set_bias_error(np.zeros(self.layers[0].get_bias_error().shape))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def train(self, inputs, epochs):
        for epoch in range(0, epochs):
            self.forward_propagation(inputs)
            self.backward_propagation(inputs)

    def test(self, input):
        activation = input
        for i in range(0, len(self.layers) - 1):
            weights = self.layers[i].thetas
            z = np.matmul(np.transpose(weights), activation)
            activation = self.sigmoid(z)

        return activation