import random

from network import Network
import numpy as np

alpha = 0.2
reg_lambda = 0.05
n = 1000
epochs = 50

network = Network([8, 3, 8], alpha, reg_lambda)
random.seed(3)

inputs = np.identity(8)
train_data = []
for i in range(0, n):
    rand_val = random.randint(0, 7)
    new_sample = inputs[rand_val]
    train_data.append(new_sample)

network.train(train_data, epochs)

for i in range(0, 8):
    print(f"Input: {inputs[i, :]}")
    result = network.test(inputs[i, :])
    print(f"Output: {result}")
