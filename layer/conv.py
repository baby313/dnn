import numpy

class Conv():
	def __init__(self, filter_count, filter_size, w, h, channel, step = 1, padding = 0):
		self.filter_count = filter_count
		self.filter_size = filter_size
		self.step = step
		self.w = w
		self.h = h
		self.padding = padding
		self.channel = channel
		self.weight = numpy.random.rand(filter_size * filter_size * self.channel, filter_count)
		x, y = self.window_count()
		self.bias = numpy.random.rand(x * y * filter_count)
	def forward(self, input):
		a = self.img2col(input)
		output = numpy.dot(a, self.weight) + self.bias
		return output
	def backward(self, output):
		intput = numpy.array([])
		return intput
	def update(self):
		return numpy.array([])
	def img2col(self, input):
		return input
	def window_count(self):
		w = self.w + self.padding * 2
		h = self.h + self.padding * 2
		x = w - self.filter_size + 1
		y = h - self.filter_size + 1
		if self.step > 1:
			d = self.filter_size - self.step if self.filter_size - self.step > 0 else 0
			x = (w - d) / self.step
			y = (h - d) / self.step
		return x, y