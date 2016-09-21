import sys
from mnist_data import Mnist
from conv import Conv
from pool import Pool
from fc import Fc
from network import Network

data = Mnist('data/mnist/train-images.idx3-ubyte', 'data/mnist/train-labels.idx1-ubyte')
conv0 = Conv(6, 5)
pool0 = Pool(2)
conv1 = Conv(6, 5)
pool1 = Pool(2)
fc0 = Fc(1024)
fc1 = Fc(1024)

net = Network()
net.add(data)
net.add(conv0, data)
#net.add(pool0, conv0)
#net.add(conv1, pool0)
#net.add(pool1, conv1)
#net.add(fc0, pool1)
#net.add(fc1, 0)

net.train()