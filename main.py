import random
from network import Network
import numpy as np

# init
alpha = 0.01
reg_lambda = 0.001
n = 10000
epochs = 300
network = Network([8, 3, 8], alpha, reg_lambda)
random.seed(3)

# create training data
inputs = np.identity(8)
train_data = []
for i in range(0, n):
    rand_val = random.randint(0, 7)
    new_sample = inputs[rand_val]
    train_data.append(new_sample)

# fit the NN
network.fit(train_data, epochs)

# test the NN
for i in range(0, 8):
    print(f"Input: {inputs[i, :]}")
    result = network.test(inputs[i, :])
    print(f"Output: {result}")
