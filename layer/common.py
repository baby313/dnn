import numpy as np

class algorithm():
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))
	def sigmoid_gradient(x):
		return (1 - x) * x

class config():
	momentum = 0.9
	decay = 0.0005
	learning_rate = 0.1
	batch = 2
	loop = 20000
	learn_rate = 0.02