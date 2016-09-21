import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))
def sigmoid_gradient(x):
	return (1 - x) * x