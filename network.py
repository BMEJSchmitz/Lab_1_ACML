import numpy as np
from layer import Layer


class Network:
    """
        Implements a NN based on the layer class.
        The main method is "fit" which uses the methods forward_pass and backward_pass to fit the NN.

        Parameters
        ----------
        layer_info: list
            the number of neurons per layer
        alpha: float
            learning rate. Determines the rate of change of the weights.
        reg_lambda: float
            regularization rate. Determines how much the regularization should modify the weights.
    """
    def __init__(self, layer_info, alpha, reg_lambda):
        self.layers = []
        self.layers.append(Layer(layer_info[0], layer_info[1], 'input'))
        self.layers.append(Layer(layer_info[1], layer_info[2], 'hidden'))
        self.layers.append(Layer(layer_info[2], 0, 'output'))

        self.alpha = alpha
        self.reg_lambda = reg_lambda

    def forward_pass(self, sample):
        """
            Performs the forward pass through the NN.
            Activations and weights are iteratively multiplied until the final layer's activation can be set.

            Parameters
            ----------
            sample : np.array
                a single input to the first layer of the NN
        """
        activation = sample.reshape(-1, 1)
        for i in range(0, len(self.layers) - 1):
            weights = self.layers[i].get_thetas()
            mul = np.matmul(weights, activation)
            bias = self.layers[i].get_bias()
            z = mul + bias
            activation = self.sigmoid(z)
            self.layers[i+1].set_activations(activation)

    def update_weights(self, m):
        """
            Updates the weights in the layer class every epoch.
            This is done according to the errors determined in the backward propagation.

            Parameters
            ----------
            m : int
                number of inputs used per epoch
        """

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
        """
            Performs backward propagation in the NN.
            Uses the activations from the forward pass to determine the error of the weights.

            Parameters
            ----------
            sample : np.array
                a single input to the first layer of the NN
        """
        activations_3 = self.layers[2].get_activations()
        z_3 = self.sigmoid_derivative(activations_3)
        small_delta_3 = np.multiply(activations_3 - sample.reshape(-1, 1), z_3)

        # hidden layer
        activations_2 = self.layers[1].get_activations()
        big_delta_2 = np.matmul(small_delta_3, activations_2.transpose())
        self.layers[1].add_error(big_delta_2)
        new_delta = self.layers[1].get_errors()
        self.layers[1].add_bias_error(small_delta_3)
        z_2 = self.sigmoid_derivative(activations_2)
        small_delta_2 = np.multiply(np.matmul(self.layers[1].get_thetas().transpose(), small_delta_3), z_2)

        # input layer
        activations_1 = sample.reshape(-1, 1)
        big_delta_1 = np.matmul(small_delta_2, activations_1.transpose())
        self.layers[0].add_error(big_delta_1)
        self.layers[0].add_bias_error(small_delta_2)

    @staticmethod
    def sigmoid(z):
        """
            Parameters
            ----------
            z : np.array
                the activations of the current layer

            Returns
            -------
            np.array
                sigmoid of activations of the layer
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        """
            Parameters
            ----------
            z : np.array
                the activations of the current layer

            Returns
            -------
            np.array
                sigmoid derivative of activations of the layer
        """
        return z * (1 - z)

    def fit(self, inputs, epochs):
        """
            Fits the weights of the NN.
            This is done via a forward and backward pass per sample where the errors are accumulated.
            These are then used for the weight update in the NN
            Done for epochs times.

            Parameters
            ----------
            inputs : np.array
                input to the NN
            epochs: int
                number of epochs to train the NN for
        """
        for epoch in range(0, epochs):
            self.layers[0].get_errors().fill(0)
            self.layers[1].get_errors().fill(0)
            self.layers[0].get_bias_error().fill(0)
            self.layers[1].get_bias_error().fill(0)

            for sample in inputs:
                self.forward_pass(sample)
                self.backward_pass(sample)

            self.update_weights(len(inputs))

    def test(self, nn_input):
        """
            Performs the forward pass on the trained NN

            Parameters
            ----------
            nn_input : np.array
                the input to the neural network

            Returns
            -------
            np.array
                activations of the final layer
        """
        activation = nn_input
        for i in range(0, len(self.layers) - 1):
            weights = self.layers[i].thetas
            z = np.matmul(weights, activation)
            activation = self.sigmoid(z)

        return activation
