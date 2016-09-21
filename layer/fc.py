import numpy as np
from common import algorithm
from common import config

class Fc():
	def __init__(self, outputs):
		self.bias = np.random.rand(outputs)
		self.bias_update = np.zeros(outputs)
		self.delta = np.zeros((config.batch, outputs))
		self.outputs = outputs
	def forward(self):
		self.output = algorithm.sigmoid(np.dot(self.prev.output, self.weight) + self.bias)
		return self.output
	def backward(self):
		self.delta *= algorithm.sigmoid_gradient(self.output)
		self.weight_update = np.dot(np.transpose(self.delta), self.prev.output)
		self.prev.delta = self.delta * self.weight
	def update(self):
		self.bias += self.bias_update
		self.weight_update -= self.weight * algorithm.decay * config.batch
		self.weight += self.weight_update * algorithm.learning_rate / config.batch
		self.bias_update *= algorithm.momentum
		self.weight_update *= algorithm.momentum
	def connected(self, prev):
		self.weight = np.random.rand(self.outputs, prev.outputs)
		self.weight_update = np.zeros((self.outputs, prev.outputs))
		self.prev = prev
