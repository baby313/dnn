import numpy as np
from common import algorithm
from common import config

class Fc():
	def __init__(self, outputs):
		self.bias = np.random.rand(outputs)
		self.bias_update = np.zeros(outputs)
		self.delta = None
		self.outputs = outputs
	def forward(self):
		self.output = algorithm.sigmoid(np.dot(self.prev.output.reshape(config.batch, -1), self.weight) + self.bias)
		return self.output
	def backward(self):
		self.delta *= algorithm.sigmoid_gradient(self.output)
		for i in self.delta:
			self.bias_update += i
		self.weight_update += np.dot(np.transpose(self.delta), self.output)
		self.prev.delta = self.delta * self.weight
	def update(self):
		self.weight_update -= self.weight * algorithm.decay * config.batch
		self.weight += self.weight_update * algorithm.learning_rate / config.batch
		self.weight_update *= algorithm.momentum
		self.bias += self.bias_update
		self.bias_update *= algorithm.momentum
	def connected(self, prev):
		self.weight = np.random.rand(prev.outputs, self.outputs)
		self.weight_update = np.zeros((prev.outputs, self.outputs))
		self.prev = prev