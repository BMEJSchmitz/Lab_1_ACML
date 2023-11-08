import random
from network import Network
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# init
alphas = [0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.90, 0.95, 1]
lambdas = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.75]
epochs = 50000
random.seed(3)

inputs = np.identity(8)

# fit and test the NN with different parameter setups
errors = []
for alpha in alphas:
    for reg_lambda in lambdas:
        print(f"Training network with alpha = {alpha} and lambda = {reg_lambda}:")
        network = Network([8, 3, 8], alpha, reg_lambda)
        history = network.fit(inputs, epochs)
        # plot training
        epoch_nums = range(0, epochs, 10)
        plt.plot(epoch_nums, history)
        plt.title(f"Training curve for alpha = {alpha} and lambda = {reg_lambda}:")
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.savefig(f'plots/curve_{alpha}_{reg_lambda}.png')
        plt.show(block=False)
        plt.clf()

        print(f"Testing network with alpha = {alpha} and lambda = {reg_lambda}:")
        for sample in inputs:
            print(f"Input: {sample}")
            result = network.test(sample)
            print(f"Output: {result.transpose()}")

        evaluation = network.evaluate(inputs)
        errors.append({"Name" : f"{alpha}_{reg_lambda}", "MSE" : evaluation})

results = pd.DataFrame(errors)
results.set_index('Name', inplace=True)
print(results)
results.to_csv("plots/results.csv")