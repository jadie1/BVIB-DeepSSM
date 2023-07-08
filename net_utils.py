# Jadie Adams
import os
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from numbers import Number
from torch.distributions.multivariate_normal import MultivariateNormal

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, w=1):
        super(UnFlatten, self).__init__()
        self.w = w

    def forward(self, x):
        return x.view((x.size(0), -1, self.w, self.w))

def poolOutDim(inDim, kernel_size, padding=0, stride=0, dilation=1):
	if stride == 0:
		stride = kernel_size
	num = inDim + 2*padding - dilation*(kernel_size - 1) - 1
	outDim = int(np.floor(num/stride + 1))
	return outDim

def cuda(tensor, is_cuda):
	if is_cuda : return tensor.cuda()
	else : return tensor 

def sample_diagonal_MultiGauss(mu, std, n):
	# reference :
	# http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
	def expand(v):
		if isinstance(v, Number):
			return torch.Tensor([v]).expand(n, 1)
		else:
			return v.expand(n, *v.size())
	if n != 1 :
		mu = expand(mu)
		std = expand(std)
	eps = Variable(std.data.new(std.size()).normal_().to(std.device))
	samples =  mu + eps * std
	samples = samples.reshape((n * mu.shape[1],)+ mu.shape[2:])
	return samples

# def sample_diagonal_MultiGauss(mu, log_var, num_samples):
# 	zs = torch.empty((num_samples, mu.shape[0], mu.shape[1]), device=mu.device, dtype=mu.dtype)
# 	for j in range(mu.shape[0]):
# 		diag_cov = torch.diag(torch.exp(log_var[j]))
# 		m = torch.distributions.MultivariateNormal(mu[j], scale_tril=diag_cov)
# 		for i in range(num_samples):
# 			zs[i,j,:] = m.rsample()
# 	return zs