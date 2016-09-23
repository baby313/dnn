import numpy as np
from common import config

class Softmax():
	def __init__(self, group):
		self.group = group
	def forward(self):
		self.input = np.pad(self.prev.output, ((0,0),(0,1)), 'constant', constant_values=((1,1),(1,1)))
		x = np.exp(np.dot(self.input, self.theta))
		self.prob = x / x.sum(axis=1).reshape(config.batch, -1)
		self.result = self.prob.argmax(axis=1)
	def backward(self):
		truth_bool = np.zeros(self.prev.output.shape)
		for i in range(config.batch):
			truth_bool[i, self.truth[i]] = 1
		self.prev.delta = (truth_bool - self.prev.output).transpose()
	def update(self):
		prob_bool = np.zeros(self.prob.shape)
		for i in range(config.batch):
			prob_bool[i, self.truth[i]] = 1
		diff = prob_bool - self.prob
		for i in range(self.group):
			self.theta_update[:, i] += (self.input * diff[:,i]).sum(axis=0).transpose()
		self.theta_update -= self.theta * algorithm.decay * config.batch
		self.theta += config.learn_rate / config.batch * self.theta_update
		self.theta_update *= config.momentum
	def connected(self, prev):
		self.prev = prev
		self.theta = np.random.rand(prev.outputs + 1, self.group)
		self.theta_update = np.zeros((prev.outputs + 1, self.group))