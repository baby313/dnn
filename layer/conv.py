import numpy
import algorithm

class Conv():
	def __init__(self, filter_count, filter_size, w, h, c, batch, step=1, padding=0):
		self.filter_count = filter_count
		self.filter_size = filter_size
		self.step = step
		self.w = w
		self.h = h
		self.padding = padding
		self.c = c
		self.batch = batch
		self.weight = numpy.random.rand(filter_size * filter_size * self.c, filter_count)
		out_w, out_h = self.conv_out_size()
		self.bias = numpy.random.rand(filter_count)
	def forward(self, input):
		return algorithm.sigmoid(numpy.dot(self.img2col(input), self.weight) + self.bias)
	def backward(self, output):
		self.bias *= algorithm.sigmoid_gradient(output)
		intput = numpy.array([])
		return intput
	def update(self):
		return numpy.array([])
	def img2col(self, input):
		input.shape = self.batch, self.c, self.h, self.w
		input = numpy.lib.pad(input, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'edge')
		batch, c, h, w = input.shape
		m = numpy.array([]).reshape(-1, self.filter_size * self.filter_size * self.c)
		for i in input:
			for y in range(0, h - self.filter_size + 1, self.step):
				for x in range(0, w - self.filter_size + 1, self.step):
					a = i[:, y:y + self.filter_size, x:x + self.filter_size].flatten()
					m = numpy.vstack((m, a))
		return m
	def conv_out_size(self):
		w = self.w + 2 * self.padding
		h = self.h + 2 * self.padding
		out_w = numpy.arange(0, w - self.filter_size + 1, self.step).shape[0]
		out_h = numpy.arange(0, h - self.filter_size + 1, self.step).shape[0]
		return out_w, out_h