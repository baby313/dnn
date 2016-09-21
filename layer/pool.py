import numpy as np
from common import config

class MaxPool():
	def __init__(self, size, step):
		self.size = size
		self.step = step
		self.delta = None
	def forward(self):
		input = prev.output.reshape(config.batch, self.c, self.h, self,w)
		self.output = array([]).reshape(-1,)
		w, h, c = self.out_size()
		for b in input:
			for c in b:
				m = np.array([]).reshape(-1, w * h)
				for i in np.arange(0, self.h - self.size, self.step):
					for j in np.arange(0, self.w - self.size, self.step):
						w = c[i:i + size, j:j + size]
						max_i, max_j = np.unravel_index(w.argmax(), w.shape)
						np.hstack(m, w.max())
				np.vstatck(self.output, m)
	def backward(self):
		w, h, c = self.output_size()
	def update(self):
		pass
	def output_size(self):
		w = np.arange(0, self.w - self.size, self.step)
		h = np.arange(0, self.h - self.size, self.step)
		return w, h, self.c
	def connected(self, prev):
		self.prev = prev
		self.w, self.h, self.c = prev.output_size()