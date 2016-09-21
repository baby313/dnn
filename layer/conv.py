import numpy as np
from common import algorithm
from common import config

class Conv():
	def __init__(self, n, size, step=1, padding=0):
		self.n = n
		self.size = size
		self.step = step
		self.padding = padding
		self.bias = np.random.rand(n).reshape(n, -1)
	def forward(self):
		input = self.prev.output.reshape(config.batch, self.c, -1)
		out_w, out_h, out_c = self.output_size()
		self.output = np.array([]).reshape(-1, out_w * out_h)
		for i in input:
			self.output = np.vstack((self.output, algorithm.sigmoid(np.dot(self.weight, self.img2col(i)) + self.bias)))
	def backward(self):
		#self.bias *= algorithm.sigmoid_gradient(self.output)
		intput = np.array([])
		return intput
	def update(self):
		return np.array([])
	def img2col(self, input):
		input.shape = self.c, self.h, self.w
		input = np.lib.pad(input, ((0,0), (self.padding, self.padding), (self.padding, self.padding)), 'edge')
		c, h, w = input.shape
		out_w, out_h, out_c = self.output_size()
		m = np.array([]).reshape(self.size * self.size * self.c, -1)
		for y in range(0, h - self.size + 1, self.step):
			for x in range(0, w - self.size + 1, self.step):
				a = input[:, y:y + self.size, x:x + self.size].flatten().reshape(self.size * self.size * self.c, -1)
				m = np.hstack((m, a))
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