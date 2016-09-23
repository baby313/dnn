import numpy as np
from common import algorithm
from common import config

class Fc():
	def __init__(self, outputs):
		self.bias = np.random.rand(outputs)
		self.bias_update = np.zeros(outputs)
		self.delta = None # shape = outputs, batch
		self.outputs = outputs
	def forward(self):
		input = self.prev.output.reshape(config.batch, self.prev.outputs)
		self.output = algorithm.sigmoid(np.dot(input, self.weight) + self.bias)
		return self.output
	def backward(self):
		self.delta = (self.delta.transpose() * algorithm.sigmoid_gradient(self.output)).transpose()
		self.bias_update += self.delta.sum(axis=1)
		self.weight_update += np.dot(self.delta, self.prev.output).transpose()
		self.prev.delta = np.dot(self.weight, self.delta)
	def update(self):
		self.weight_update += self.weight * config.decay * config.batch
		self.weight += self.weight_update * config.learning_rate / config.batch
		self.weight_update *= config.momentum
		self.bias += self.bias_update * config.learning_rate / config.batch
		self.bias_update *= config.momentum
	def connected(self, prev):
		self.weight = np.random.rand(prev.outputs, self.outputs)
		self.weight_update = np.zeros((prev.outputs, self.outputs))
		self.prev = prev