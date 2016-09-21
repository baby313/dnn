import numpy as np
import algorithm

class Fc():
	def __init__(self, batch, inputs, outputs):
		self.batch = batch
		self.weight = np.random.rand(inputs * outputs)
		self.bias = np.random.rand(outputs)
	def forward(self, input):
		return algorithm.sigmoid(np.dot(input, self.weight) + self.bias)
	def backward(self, output):
		self.bias *= algorithm.sigmoid_gradient(output)
		intput = np.array([])
		return intput
	def update(self):
		return np.array([])