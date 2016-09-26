import numpy as np
from common import config

class MaxPool():
	def __init__(self, size, step):
		self.size = size
		self.step = step
		self.delta = None
	def forward(self):
		input = self.prev.output.reshape(config.batch, self.c, self.h, self.w)
		self.output = np.zeros((config.batch, self.c, self.h // self.step, self.w // self.step))
		out_w, out_h, c = self.output_size()
		self.idx = np.zeros((config.batch, self.c, self.h // self.step, self.w // self.step, 2))
		for b in range(config.batch):
			for c in range(self.c):
				for h in range(out_h):
					for w in range(out_w):
						i = h * self.step
						j = w * self.step
						x = input[b, c, i:i + self.size, j:j + self.size]
						self.output[b, c, h, w] = x.max()
						self.idx[b, c, h, w] = np.unravel_index(x.argmax(), x.shape)
		self.output.shape = config.batch, -1
	def backward(self):
		self.prev.delta = np.zeros((config.batch, self.c, self.h, self.w))
		out_w, out_h, c = self.output_size()
		delta = self.delta.reshape(config.batch, c, out_h, out_w)
		for b in range(config.batch):
			for c in range(self.c):
				for h in range(out_h):
					for w in range(out_w):
						max_i, max_j = self.idx[b, c, h, w]
						self.prev.delta[np.int8(max_i), np.int8(max_j)] = delta[b, c, h, w]
		self.prev.delta.shape = config.batch, -1
	def update(self):
		pass
	def output_size(self):
		w = np.arange(0, self.w - self.size + 1, self.step).shape[0]
		h = np.arange(0, self.h - self.size + 1, self.step).shape[0]
		return w, h, self.c
	def connected(self, prev):
		self.prev = prev
		self.w, self.h, self.c = prev.output_size()
		out_w, out_h, c = self.output_size()
		self.outputs = out_w * out_h * self.c