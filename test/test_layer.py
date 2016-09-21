from conv import Conv
from pool import Pool
from fc import Fc
from mnist_data import Mnist
import numpy as np

def test_conv():
	data = Mnist('data/mnist/train-images.idx3-ubyte', 'data/mnist/train-labels.idx1-ubyte')
	conv = Conv(2, 3)
	conv.connected(data)
	data.forward()
	conv.forward()

def test_fc():
	fc0 = Fc(256)
	fc1 = Fc(1024)
	fc1.connected(fc0)
	fc1.forward()

test_conv()