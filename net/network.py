from common import config

class Network():
	def __init__(self):
		self.layers = []
	def add(self, layer, prev=None):
		self.layers.append(layer)
		if prev != None:
			layer.connected(prev)
	def forward(self):
		for l in self.layers[:-1]:
			l.forward()
	def backward(self):
		for l in self.layers[:0:-1]:
			l.backward()
	def update(self):
		for l in self.layers:
			l.update()
	def train(self):
		for i in range(config.loop):
			self.forward()
			self.backward()
			self.update()
			#print(i)