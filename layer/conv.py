import numpy as np
from common import algorithm
from common import config

class Conv():
	def __init__(self, n, size, step=1, padding=0):
		self.n = n
		self.size = size
		self.step = step
		self.padding = padding
		self.delta = None
	def forward(self):
		input = self.prev.output.reshape(config.batch, self.c, -1)
		out_w, out_h, out_c = self.output_size()
		self.output = np.zeros((config.batch, out_c, out_w * out_h))
		for i in range(config.batch):
			self.output[i] = algorithm.sigmoid(np.dot(self.weight, self.img2col(input[i])) + self.bias)
	def backward(self):
		input = self.prev.output.reshape(config.batch, self.c, -1)
		self.delta *= algorithm.sigmoid_gradient(self.output.reshape(config.batch, -1))
		self.bias_update += self.delta.sum(axis=0).reshape(self.n, -1)
		self.prev.delta = np.zeros((config.batch, self.c * self.h * self.w))
		for i in range(config.batch):
			self.weight_update += np.dot(self.delta[i].reshape(self.n, -1), self.img2col(input[i]).transpose())
			prev_delta = np.dot(self.weight.transpose(), self.delta[i].reshape(self.n, -1))
			self.prev.delta[i] = self.col2img(prev_delta).reshape(1, -1)
	def update(self):
		self.bias += self.bias_update * config.learn_rate / config.batch
		self.bias_update *= config.momentum
		self.weight_update -= self.weight * config.decay * config.batch
		self.weight += self.weight_update * config.learn_rate / config.batch
		self.weight_update *= config.momentum
	def img2col(self, input): # from (c, h * w) to (c * size * size, out_h * out_w)
		input.shape = self.c, self.h, self.w
		input = np.lib.pad(input, ((0,0), (self.padding, self.padding), (self.padding, self.padding)), 'edge')
		c, h, w = input.shape
		out_w, out_h, out_c = self.output_size()
		m = np.zeros((self.size * self.size * self.c, out_w * out_h))
		i = 0
		for y in range(0, h - self.size + 1, self.step):
			for x in range(0, w - self.size + 1, self.step):
				a = input[:, y:y + self.size, x:x + self.size].flatten()
				m[:, i] = a
				i += 1
		return m
	def col2img(self, input): # from (c * size * size, out_h * out_w) to (c, h * w)
		m = np.zeros((self.c, self.h + self.padding * 2, self.w + self.padding * 2))
		c, h, w = m.shape
		i = 0
		for y in range(0, h - self.size + 1, self.step):
			for x in range(0, w - self.size + 1, self.step):
				m[:, y:y + self.size, x:x + self.size] = input[:, i].reshape(c, self.size, self.size)
				i += 1
		return m
	def output_size(self):
		w = self.w + 2 * self.padding
		h = self.h + 2 * self.padding
		out_w = np.arange(0, w - self.size + 1, self.step).shape[0]
		out_h = np.arange(0, h - self.size + 1, self.step).shape[0]
		return out_w, out_h, self.n
	def connected(self, prev):
		self.prev = prev
		self.w, self.h, self.c = prev.output_size()
		self.weight = np.random.rand(self.n, self.size * self.size * self.c)
		self.weight_update = np.zeros((self.n, self.size * self.size * self.c))
		out_w, out_h, out_c = self.output_size()
		self.bias = np.random.rand(self.n, out_h * out_w)
		self.bias_update = np.zeros((self.n, out_h * out_w))