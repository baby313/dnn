from net.lenet import *

img_count, w, h, img_arr = read_images('data/mnist/train-images.idx3-ubyte')
lbl_count, lbl_arr = read_labels('data/mnist/train-labels.idx1-ubyte')

print img_arr.shape