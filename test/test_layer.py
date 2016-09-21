import numpy
from conv import Conv
from pool import Pool

def test_conv():
	input = numpy.array([1, 2, 3]).repeat(25)
	conv = Conv(2, 3, 5, 5, 3, 1)
	conv.forward(input)
	
test_conv()
#test_pool()
#test_fc()
#test_softmax()