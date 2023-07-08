import torch
import torch.distributions as dist
import math
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
epsilon=1e-6 

## Loss functions - same arguements for consistency in training loop
def pca_mse(pred_z, true_z, pred_y, true_y, params={}):
	predicted_z = pred_z[0]
	return MSE(predicted_z, true_z)

def corr_mse(pred_z, true_z, pred_y, true_y, params={}):
	predicted_y = pred_y[0]
	loss = MSE(predicted_y, true_y)
	return loss

def pca_nll(pred_z, true_z, pred_y, true_y, params={}):
	z_mu = pred_z[0]
	z_log_sigma = pred_z[1]
	loss = NLL(z_mu, z_log_sigma, true_z)
	return loss

def corr_nll(pred_z, true_z, pred_y, true_y, params={}):
	y_mu = pred_y[0]
	y_log_sigma = pred_y[1].to(y_mu.device)
	# loss_func = torch.nn.GaussianNLLLoss()
	# loss = loss_func(y_mu, true_y, torch.exp(y_log_sigma))
	loss = NLL(y_mu, y_log_sigma, true_y)
	return loss

def pca_nll_burnin(pred_z, true_z, pred_y, true_y, params={}):
	z_mu = pred_z[0]
	z_log_sigma = pred_z[1]
	epoch = params["epoch"]
	initiate = params['initiate_stochastic']
	comp = params['complete_stochastic']
	y_mse = MSE(pred_y[0], true_y)	
	# Deterministic phase
	if epoch <= initiate:
		loss = y_mse
	# Introduce stochastic
	else:
		alpha = min(1, ((epoch - initiate)/(comp - initiate)))
		z_nll = NLL(z_mu, z_log_sigma, true_z)
		loss = (1-alpha)*y_mse + alpha*z_nll
	return loss

def ppca_offset(pred_z, true_z, pred_y, true_y, params):
	z_mean = pred_z[0]
	z_log_var = pred_z[1]
	offset = pred_z[2]
	zs = pred_z[3]
	y_mean = pred_y[0]
	y_log_var = pred_y[1]
	zs.to(z_mean.device)
	mix = dist.Categorical(torch.ones(z_mean.shape[0],).to(z_mean.device))
	comp = dist.Independent(dist.Normal(z_mean, torch.exp(0.5*(z_log_var + epsilon))), 1)
	z_dist = dist.MixtureSameFamily(mix, comp)
	prior  = dist.Normal(torch.FloatTensor([0.]).to(z_mean.device), torch.FloatTensor([1.]).to(z_mean.device))
	pred_log_prob = z_dist.log_prob(zs)
	prior_log_prob = torch.sum(prior.log_prob(zs), axis=2)
	kld = torch.sum(pred_log_prob - prior_log_prob)
	y_nll = torch.mean(NLL(y_mean + offset, y_log_var.to(y_mean), true_y))
	loss = y_nll + lam*kld
	return loss

def ppca_offset_burnin(pred_z, true_z, pred_y, true_y, params):
	z_mean = pred_z[0]
	z_log_var = pred_z[1]
	offset = pred_z[2]
	zs = pred_z[3]
	y_mean = pred_y[0]
	y_log_var = pred_y[1]
	epoch = params["epoch"]
	lam = params['lambda']
	initiate = params['initiate_stochastic']
	comp = params['complete_stochastic']
	y_mse = MSE(y_mean+offset, true_y)	
	# Deterministic phase
	if epoch <= initiate:
		loss = y_mse
	# Introduce stochastic
	else:
		alpha = min(1, ((epoch - initiate)/(comp - initiate)))
		zs.to(z_mean.device)
		mix = dist.Categorical(torch.ones(z_mean.shape[0],).to(z_mean.device))
		comp = dist.Independent(dist.Normal(z_mean, torch.exp(0.5*(z_log_var + epsilon))), 1)
		z_dist = dist.MixtureSameFamily(mix, comp)
		prior  = dist.Normal(torch.FloatTensor([0.]).to(z_mean.device), torch.FloatTensor([1.]).to(z_mean.device))
		pred_log_prob = z_dist.log_prob(zs)
		prior_log_prob = torch.sum(prior.log_prob(zs), axis=2)
		kld = torch.sum(pred_log_prob - prior_log_prob)
		y_nll = torch.mean(NLL(y_mean + offset, y_log_var.to(y_mean), true_y))
		loss = (1-alpha)*y_mse + alpha*y_nll + alpha*lam*kld
	return loss

def vib(pred_z, true_z, pred_y, true_y, params):
	beta = params['beta']
	y_nll = corr_nll(pred_z, true_z, pred_y, true_y)
	z_kld = KLD(pred_z[0], pred_z[1])
	loss = y_nll + beta*z_kld
	return loss

def vib_burnin(pred_z, true_z, pred_y, true_y, params):
	epoch = params["epoch"]
	beta = params['beta']
	init = params['initiate_stochastic']
	comp =params['complete_stochastic']
	y_mse = corr_mse(pred_z, true_z, pred_y, true_y)
	y_nll = corr_nll(pred_z, true_z, pred_y, true_y)
	z_kld = KLD(pred_z[0], pred_z[1])
	# Deterministic phase
	if epoch <= init:
		loss = y_mse
	# Introduce stochastic
	else:
		alpha = min(1, ((epoch - init)/(comp - init)))
		loss = (1-alpha)*y_mse + alpha*y_nll + alpha*beta*z_kld
	return loss

####### Helper functions

def MSE(predicted, true):
	return torch.mean((predicted - true)**2)

def NLL(mu, log_sigma, ground_truth):
	log_sigma = log_sigma + epsilon
	nll_loss = 0.5 * (log_sigma + (mu - ground_truth)**2 / torch.exp(log_sigma)) # + 0.5 * math.log(2 * math.pi)
	return torch.mean(nll_loss)

# Same as KLD
def KLD2(mu, log_sigma):
	prior  = dist.Normal(torch.FloatTensor([0.]).to(mu.device), torch.FloatTensor([1.]).to(mu.device))
	kld = dist.kl_divergence(dist.Normal(mu, torch.exp(0.5*(log_sigma))), prior)
	return torch.mean(kld)

def KLD(mu, log_sigma):
	log_sigma = log_sigma + epsilon
	# kld = -0.5 * torch.sum(1 + z_log_sigma - z_mu.pow(2) - z_log_sigma.exp(), dim=1) 
	kld = -0.5 * (1 + log_sigma - mu.pow(2) - (log_sigma).exp()) 
	return torch.mean(kld)
