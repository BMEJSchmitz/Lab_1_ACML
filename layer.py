import numpy as np
import random


class Layer:

    def __init__(self, neurons, previous_neurons):
        random.seed(3)
        self.neurons = neurons + 1  # int
        self.thetas = (np.random.rand(neurons, previous_neurons + 1) / 2)+0.000001  # weights (2d vector)/matrix
        self.activations = np.zeros(neurons)  # (1d vector)

    def get_thetas(self):
        return self.thetas

    def get_activations(self):
        return self.activations

    def set_thetas(self, new_thetas):
        self.thetas = new_thetas

    def set_activations(self, new_activations):
        self.activations = new_activations
