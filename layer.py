import numpy as np
import random


class Layer:
    """
        Implements the layer in an NN.
        Store attributes thetas, activations, error, bias, bias_error.

        Parameters
        ----------
        neurons: int
            number of neurons
        next_neurons: int
            number of neurons in the next layer
        flag: str
            The kind of layer: input, hidden or output
    """

    def __init__(self, neurons, next_neurons, flag):
        small_constant = 0.000001
        random.seed(3)
        self.flag = flag
        self.neurons = neurons  # int
        if flag != 'input':
            self.activations = np.zeros(neurons)  # (1d vector)
        if flag != 'output':
            self.thetas = (np.random.rand(next_neurons, neurons) / 2)+small_constant  # weights (2d vector)/matrix
            self.errors = np.zeros((next_neurons, neurons))
            self.bias = (np.random.rand(next_neurons, 1) / 2)+small_constant
            self.bias_error = np.zeros((next_neurons, 1))

    def get_thetas(self):
        return self.thetas

    def get_activations(self):
        return self.activations

    def get_errors(self):
        return self.errors

    def get_bias(self):
        return self.bias

    def get_bias_error(self):
        return self.bias_error

    def set_thetas(self, new_thetas):
        self.thetas = new_thetas

    def set_activations(self, new_activations):
        self.activations = new_activations

    def set_errors(self, new_errors):
        self.errors = new_errors

    def set_bias(self, new_bias):
        self.bias = new_bias

    def set_bias_error(self, new_bias_error):
        self.bias_error = new_bias_error

    def add_bias_error(self, error):
        self.bias_error += error

    def add_error(self, error):
        self.errors += error
