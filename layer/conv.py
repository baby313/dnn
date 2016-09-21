import numpy
from common import algorithm
from common import config

class Conv():
	def __init__(self, n, size, step=1, padding=0):
		self.n = n
		self.size = size
		self.step = step
		self.padding = padding
		self.bias = numpy.random.rand(n)
	def forward(self):
		self.output = algorithm.sigmoid(numpy.dot(self.img2col(self.prev.output), self.weight) + self.bias)
	def backward(self):
		#self.bias *= algorithm.sigmoid_gradient(self.output)
		intput = numpy.array([])
		return intput
	def update(self):
		return numpy.array([])
	def img2col(self, input):
		input.shape = config.batch, self.c, self.h, self.w
		input = numpy.lib.pad(input, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'edge')
		batch, c, h, w = input.shape
		m = numpy.array([]).reshape(-1, self.size * self.size * self.c)
		for i in input:
			for y in range(0, h - self.size + 1, self.step):
				for x in range(0, w - self.size + 1, self.step):
					a = i[:, y:y + self.size, x:x + self.size].flatten()
					m = numpy.vstack((m, a))
		return m
	def output_size(self):
		w = self.w + 2 * self.padding
		h = self.h + 2 * self.padding
		out_w = numpy.arange(0, w - self.size + 1, self.step).shape[0]
		out_h = numpy.arange(0, h - self.size + 1, self.step).shape[0]
		return out_w, out_h, self.size
	def connected(self, prev):
		self.prev = prev
		self.w, self.h, self.c = prev.output_size()
		self.weight = numpy.random.rand(self.size * self.size * self.c, self.n)