import random

from network import Network
import numpy as np

alpha = 0.5
reg_lambda = 0.5
n = 10000
network = Network([8, 3, 8], alpha, reg_lambda)

inputs = np.identity(8)
train_data = np.array(n)
for i in range(0, n):
    rand_val = random.randint(0, 7)
    new_sample = inputs[rand_val]
    train_data[i] = new_sample

network.train(train_data)
