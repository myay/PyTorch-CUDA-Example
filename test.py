import math
from torch import nn
from torch.autograd import Function
import torch

import test_cuda

X = torch.cuda.FloatTensor(1024).uniform_()
print(X)
Xout = test_cuda.test(X)
print(Xout)
# call cuda kernel to add 1 to every element
