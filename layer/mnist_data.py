import numpy
import struct
from common import config

class Mnist():
	def __init__(self, img_filename, lbl_filename):
		self.img_count, self.w, self.h, self.images = self.read_images(img_filename)
		self.lbl_count, self.labels = self.read_labels(lbl_filename)
		self.images.shape = self.img_count, self.w * self.h
	def read_images(self, filename):
		f = open(filename, 'rb')
		buf = f.read()
		magic, n, w, h = struct.unpack_from('>IIII', buf, 0)
		if magic != 2051:
			raise IOError('not a correct mnist image file')
		images = numpy.frombuffer(buf, dtype=numpy.uint8, offset=16)
		f.close()
		return n, w, h, images
	def read_labels(self, filename):
		f = open(filename, 'rb')
		buf = f.read()
		magic, n = struct.unpack_from('>II', buf, 0)
		if magic != 2049:
			raise IOError('not a correct mnist image file')
		labels = numpy.frombuffer(buf, dtype=numpy.uint8, offset=8)
		f.close()
		return n, labels
	def forward(self):
		self.choice = numpy.random.choice(self.img_count, config.batch)
		self.output = self.images[self.choice]
	def backward(self):
		self.prev.truth = self.labels[self.choice]
	def update(self):
		pass
	def output_size(self):
		return self.w, self.h, 1
	def connected(self, prev):
		self.prev = prev