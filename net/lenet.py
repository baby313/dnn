import sys
from mnist_data import Mnist
from conv import Conv
from pool import MaxPool
from fc import Fc
from network import Network
from softmax import Softmax

data = Mnist('data/mnist/train-images.idx3-ubyte', 'data/mnist/train-labels.idx1-ubyte')
conv0 = Conv(20, 5)
pool0 = MaxPool(2, 2)
conv1 = Conv(50, 5)
pool1 = MaxPool(2, 2)
fc0 = Fc(500)
fc1 = Fc(10)
softmax0 = Softmax(10)

net = Network()
net.add(data)
net.add(conv0, data)
net.add(pool0, conv0)
net.add(conv1, pool0)
net.add(pool1, conv1)
net.add(fc0, pool1)
net.add(fc1, fc0)
net.add(softmax0, fc1)
net.add(data, softmax0)

net.train()