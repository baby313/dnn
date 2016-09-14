import sys
sys.path.append("..")
from layer.mnist_data import Mnist
from layer.conv import Conv
from layer.pool import Pool

batch = 2
learn_rate = 0.02
itration_count = 50000

train_data = Mnist('../data/mnist/train-images.idx3-ubyte', '../data/mnist/train-labels.idx1-ubyte', batch)
net = [Conv(6, 5, 28, 28, 3), Pool(2)]

def train():
	for itr in xrange(itration_count):
		for b in xrange(batch):
			forward()
			backward()
		update()
def forward():
	data = train_data.get_input()
	for layer in net:
		data = layer.forward(data)
def backward():
	data = train_data.get_output()
	for layer in net[::-1]:
		data = layer.backward(data)
def update():
	return None
	
train()