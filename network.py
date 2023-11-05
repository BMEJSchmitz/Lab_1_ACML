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

    def forward_pass(self, sample):
        activation = sample.reshape(-1, 1)
        for i in range(0, len(self.layers) - 1):
            weights = self.layers[i].get_thetas()
            mul = np.matmul(weights, activation)
            bias = self.layers[i].get_bias()
            z = mul + bias
            activation = self.sigmoid(z)
            self.layers[i+1].set_activations(activation)

    def update_weights(self, m):
        theta_update = (1 / m) * self.layers[1].get_errors()
        reg = self.reg_lambda * self.layers[1].get_thetas()
        new_theta = self.layers[1].get_thetas() - (self.alpha * (theta_update + reg))
        self.layers[1].set_thetas(new_theta)

        bias_update = (1 / m) * self.layers[1].get_bias_error()
        new_bias = self.layers[1].get_bias() - (self.alpha * bias_update)
        self.layers[1].set_bias(new_bias)

        theta_update = (1 / m) * self.layers[0].get_errors()
        reg = self.reg_lambda * self.layers[0].get_thetas()
        new_theta = self.layers[0].get_thetas() - (self.alpha * (theta_update + reg))
        self.layers[0].set_thetas(new_theta)

        bias_update = (1 / m) * self.layers[0].get_bias_error()
        new_bias = self.layers[0].get_bias() - (self.alpha * bias_update)
        self.layers[0].set_bias(new_bias)

    def backward_pass(self, sample):
        activations = self.layers[2].get_activations()
        z = self.sigmoid_derivative(activations)
        small_delta = np.multiply(activations - sample.reshape(-1, 1), z)

        # hidden layer
        activations = self.layers[1].get_activations()
        big_delta = np.matmul(small_delta, activations.transpose())
        self.layers[1].add_error(big_delta)
        new_delta = self.layers[1].get_errors()
        self.layers[1].add_bias_error(small_delta)
        z = self.sigmoid_derivative(activations)
        small_delta = np.multiply(np.matmul(self.layers[1].get_thetas().transpose(), small_delta), z)

        # input layer
        activations = sample.reshape(-1, 1)
        big_delta = np.matmul(small_delta, activations.transpose())
        self.layers[0].add_error(big_delta)
        self.layers[0].add_bias_error(small_delta)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def fit(self, inputs, epochs):
        for epoch in range(0, epochs):
            self.layers[0].get_errors().fill(0)
            self.layers[1].get_errors().fill(0)
            self.layers[0].get_bias_error().fill(0)
            self.layers[1].get_bias_error().fill(0)

            for sample in inputs:
                self.forward_pass(sample)
                self.backward_pass(sample)

            self.update_weights(len(inputs))

    def test(self, input):
        activation = input
        for i in range(0, len(self.layers) - 1):
            weights = self.layers[i].thetas
            z = np.matmul(weights, activation)
            activation = self.sigmoid(z)

        return activation