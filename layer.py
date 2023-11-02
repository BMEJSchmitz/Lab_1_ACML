import numpy as np
import random


class Layer:

    def __init__(self, neurons, next_neurons, flag):
        random.seed(3)
        self.flag = flag
        self.neurons = neurons + 1  # int
        if flag is not 'input':
            self.activations = np.zeros(neurons)  # (1d vector)
        if flag is not 'output':
            self.thetas = (np.random.rand(neurons + 1, next_neurons) / 2)+0.000001  # weights (2d vector)/matrix
            self.errors = np.zeros((neurons + 1, next_neurons))

    def get_thetas(self):
        return self.thetas

    def get_activations(self):
        return self.activations

    def get_errors(self):
        return self.errors

    def set_thetas(self, new_thetas):
        self.thetas = new_thetas

    def set_activations(self, new_activations):
        self.activations = new_activations

    def set_errors(self, new_errors):
        self.errors = new_errors
