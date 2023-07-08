import torch
from torch import nn
import torch.nn.functional as F
import sys
import math
import json
import numpy as np
from collections import OrderedDict
import net_utils
from torch.autograd import Variable
from numbers import Number
# import torch.distributions as dist
from torch.distributions.multivariate_normal import MultivariateNormal
from layers import *

'''
BayesianDeepSSM
'''
class BayesianDeepSSMNet(nn.Module):
	def __init__(self, config_file):
		super(BayesianDeepSSMNet, self).__init__()
		if torch.cuda.is_available():
			device = 'cuda:0'
		else:
			device = 'cpu'
		self.device = device
		with open(config_file) as json_file: 
			config = json.load(json_file)
		self.config = config

		# Set dimensions
		loader = torch.load(config['paths']['loader_dir'] + "validation")
		self.y_dim = loader.dataset.mdl_target[0].shape[0]*3
		self.x_dim = loader.dataset.img[0].shape[1:]
		self.z_dim = config['num_latent_dim']
		self.dropout = config['dropout']
		self.batch_ensemble = config['batch_ensemble']
		# Encoder output dim
		if not config['encoder']['stochastic']:
			self.z_dist_dim = self.z_dim
		else:
			# Set z distribution dimension
			self.z_dist_dim = self.z_dim
			if config['encoder']['covariance_type']=='diagonal':
				self.z_dist_dim += self.z_dim 
			elif config['encoder']['covariance_type']=='lower_tri':
				self.z_dist_dim += (self.z_dim*(self.z_dim+1)//2)
			elif config['encoder']['covariance_type']=='full':
				self.z_dist_dim += (self.z_dim*self.z_dim)
		
		# Set encoder
		self.encoder = Encoder(self.x_dim, self.z_dist_dim, self.dropout, self.batch_ensemble)
		
		# Set decoder
		self.decoder = NonLinearDecoder(self.z_dim, self.y_dim, self.dropout, self.batch_ensemble)

	def forward(self, x, num_samples=1, use_dropout=False):
		
		''' Encode '''
		z, encoder_reg = self.encoder(x, use_dropout)

		# Deterministic z
		if not self.config['encoder']['stochastic']:
			z_mean = z
			z_log_var = torch.zeros(z_mean.size()) # placeholder
		# Stochastic z
		else:
			z_mean = z[:,:self.z_dim]
			z_log_var = z[:,self.z_dim:]

		''' Decode '''

		# Deterministic z
		if not self.config['encoder']['stochastic']:
			y_mean, decoder_reg = self.decoder(z_mean, use_dropout)
			y_log_var = torch.zeros(y_mean.size()) # placeholder 
		# Stochastic z
		else:
			# If sampling off (test mode)
			if num_samples==0: 
				y_mean, decoder_reg = self.decoder(z_mean, use_dropout)
				y_log_var = torch.zeros(y_mean.size()) # placeholder
			# If sampling on 
			else:
				if self.config['encoder']['covariance_type']=='diagonal':
					zs = net_utils.sample_diagonal_MultiGauss(z_mean, z_log_var, num_samples)
				elif self.config['encoder']['covariance_type']=='lower_tri':
					pass
				elif self.config['encoder']['covariance_type']=='full':
					pass
				# Decode 
				ys, decoder_reg = self.decoder(zs, use_dropout) 
				ys = ys.reshape(num_samples, x.shape[0], ys.shape[1])
				y_mean = ys.mean(0)
				y_log_var = torch.log(ys.var(0))

		return [z_mean, z_log_var], [y_mean, y_log_var], (encoder_reg+decoder_reg)

class ConvolutionalBackbone(nn.Module):
    def __init__(self, x_dim, dropout={"type":None}, batch_ensemble={"enabled":False}):
        super(ConvolutionalBackbone, self).__init__()
        self.x_dim = x_dim
        # basically using the number of dims and the number of poolings to be used 
        # figure out the size of the last fc layer so that this network is general to 
        # any images
        self.out_fc_dim = np.copy(x_dim)
        padvals = [4, 8, 8]
        for i in range(3):
            self.out_fc_dim[0] = net_utils.poolOutDim(self.out_fc_dim[0] - padvals[i], 2)
            self.out_fc_dim[1] = net_utils.poolOutDim(self.out_fc_dim[1] - padvals[i], 2)
            self.out_fc_dim[2] = net_utils.poolOutDim(self.out_fc_dim[2] - padvals[i], 2)
        self.conv_out_dim = self.out_fc_dim[0]*self.out_fc_dim[1]*self.out_fc_dim[2]*192
        self.fc_out_dim = int(self.conv_out_dim*.05)
        self.fc_out_dim = int(self.conv_out_dim*.02)
        self.final_dim = int(self.conv_out_dim*.01)

        self.dropout_type = dropout['type']
        if batch_ensemble['enabled']:
            self.batch_ensemble_num_models = batch_ensemble['num_models']
            self.mixup = batch_ensemble['mixup']
        else:
            self.batch_ensemble_num_models = 0
            self.mixup = False

        # Set convolution blocks
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(Conv3d_Block( 1, 12, 5, batch_norm=True, max_pool=True,  be_num_models=self.batch_ensemble_num_models, be_mixup=self.mixup))
        self.conv_blocks.append(Conv3d_Block(12, 24, 5, batch_norm=True, max_pool=False, be_num_models=self.batch_ensemble_num_models, be_mixup=self.mixup))
        self.conv_blocks.append(Conv3d_Block(24, 48, 5, batch_norm=True, max_pool=True,  be_num_models=self.batch_ensemble_num_models, be_mixup=self.mixup))
        self.conv_blocks.append(Conv3d_Block(48, 96, 5, batch_norm=True, max_pool=False, be_num_models=self.batch_ensemble_num_models, be_mixup=self.mixup))
        self.conv_blocks.append(Conv3d_Block(96,192, 5, batch_norm=True, max_pool=True,  be_num_models=self.batch_ensemble_num_models, be_mixup=self.mixup))

        # Set conv 3D dropout
        self.conv_dropouts = nn.ModuleList()
        for i in range(5):
            if self.dropout_type=="MC":
                self.conv_dropouts.append(nn.Dropout3d(dropout["params"]["rate"]))
            elif self.dropout_type=="concrete":
                weight_reg = dropout["params"]["lengthscale"]**2./dropout["params"]["size"] 
                drop_reg = 2./(dropout["params"]["size"]*1000)
                self.conv_dropouts.append(SpatialConcreteDropout(weight_reg, drop_reg, dropout["params"]["init_rate"], dropout["params"]["init_rate"]))

        # Set fully connected blocks
        self.fc_blocks = nn.ModuleList()
        self.fc_blocks.append(FC_Block(self.conv_out_dim, self.fc_out_dim, flatten=True, be_num_models=self.batch_ensemble_num_models, be_mixup=self.mixup))
        self.fc_blocks.append(FC_Block(self.fc_out_dim,   self.final_dim, flatten=False, be_num_models=self.batch_ensemble_num_models, be_mixup=self.mixup))
           
        # Set fc 1d dropouts
        self.fc_dropouts = nn.ModuleList()
        for i in range(2):
            if self.dropout_type=="MC":
                self.fc_dropouts.append(nn.Dropout(dropout["params"]["rate"]))
            elif self.dropout_type=="concrete":
                weight_reg = dropout["params"]["lengthscale"]**2./dropout["params"]["size"] 
                drop_reg = 2./(dropout["params"]["size"]*1000)
                self.fc_dropouts.append(ConcreteDropout(weight_reg, drop_reg, dropout["params"]["init_rate"], dropout["params"]["init_rate"]))

    def forward(self, x, use_dropout):
        # Regularization is 0 unless dropout is concrete
        regularization = torch.tensor(0, device=x.device).type(x.dtype) # placeholder   
        if self.dropout_type is None or use_dropout is False:
            for i in range(5):
                x = self.conv_blocks[i](x)
            for i in range(2):
                x = self.fc_blocks[i](x)
        elif self.dropout_type=='concrete':
            regularization = torch.empty(7, device=x.device, dtype=x.dtype)
            for i in range(5):
                x, regularization[i] = self.conv_dropouts[i](x, self.conv_blocks[i])
            for i in range(2):
                x, regularization[i+5] = self.fc_dropouts[i](x, self.fc_blocks[i])
            regularization = regularization.sum()
        else:
            for i in range(5):
            	x = self.conv_dropouts[i](self.conv_blocks[i](x))
            for i in range(2):
                x = self.fc_dropouts[i](self.fc_blocks[i](x))
        return x, regularization

class Encoder(nn.Module):
	def __init__(self, x_dim, z_dist_dim, dropout={"type":None}, batch_ensemble={"enabled":False}):
		super(Encoder, self).__init__()
		self.ConvolutionalBackbone = ConvolutionalBackbone(x_dim, dropout, batch_ensemble)
		if batch_ensemble["enabled"]:
			self.pred_z_dist = BatchEnsemble_orderFC(self.ConvolutionalBackbone.final_dim, z_dist_dim, \
				num_models=batch_ensemble["num_models"], mixup=batch_ensemble["mixup"])
		else:
			self.pred_z_dist = nn.Linear(self.ConvolutionalBackbone.final_dim, z_dist_dim)
	def forward(self, x, use_dropout):
		features, regularization = self.ConvolutionalBackbone(x, use_dropout)
		z_dist = self.pred_z_dist(features)
		return z_dist, regularization

class NonLinearDecoder(nn.Module):
	def __init__(self, z_dim, y_dim, dropout={"type":None}, batch_ensemble={"enabled":False}):
		super(NonLinearDecoder, self).__init__()
		self.dropout_rate = 0.2
		self.z_dim = z_dim
		self.y_dim = y_dim
		self.mid_dim1 = int((z_dim+y_dim)/3)
		self.mid_dim2 = 2*self.mid_dim1
		self.dropout_type = dropout['type']
		if batch_ensemble['enabled']:
			self.batch_ensemble_num_models = batch_ensemble['num_models']
			self.mixup = batch_ensemble['mixup']
		else:
			self.batch_ensemble_num_models = 0
			self.mixup = False

		# Set fully connected blocks
		self.fc_blocks = nn.ModuleList()
		self.fc_blocks.append(FC_Block(self.z_dim, int((z_dim+y_dim)/4), flatten=False, be_num_models=self.batch_ensemble_num_models, be_mixup=self.mixup))
		self.fc_blocks.append(FC_Block(int((z_dim+y_dim)/4), int((z_dim+y_dim)/2), flatten=False, be_num_models=self.batch_ensemble_num_models, be_mixup = self.mixup))
		self.fc_blocks.append(FC_Block(int((z_dim+y_dim)/2), int(3*(z_dim+y_dim)/4), flatten=False, be_num_models=self.batch_ensemble_num_models, be_mixup = self.mixup))

		# Set fc 1d dropouts
		self.fc_dropouts = nn.ModuleList()
		for i in range(3):
			if self.dropout_type=="MC":
				self.fc_dropouts.append(nn.Dropout(dropout["params"]["rate"]))
			elif self.dropout_type=="concrete":
				weight_reg = dropout["params"]["lengthscale"]**2./dropout["params"]["size"] 
				drop_reg = 2./(dropout["params"]["size"]*1000)
				self.fc_dropouts.append(ConcreteDropout(weight_reg, drop_reg, dropout["params"]["init_rate"], dropout["params"]["init_rate"]))

		if batch_ensemble["enabled"]:
			self.pred_y = BatchEnsemble_orderFC(int(3*(z_dim+y_dim)/4), self.y_dim, \
				num_models=batch_ensemble["num_models"], mixup=batch_ensemble["mixup"])
		else:
			self.pred_y = nn.Linear(int(3*(z_dim+y_dim)/4), self.y_dim)
	def forward(self, z, use_dropout):
		# Regularization is 0 unless dropout is concrete
		regularization = torch.tensor(0, device=z.device).type(z.dtype) # placeholder		
		if self.dropout_type is None or use_dropout is False:
			for i in range(3):
				z = self.fc_blocks[i](z)
		elif self.dropout_type=='concrete':
			regularization = torch.empty(3, device=z.device, dtype=z.dtype)
			for i in range(3):
				z, regularization[i] = self.fc_dropouts[i](z, self.fc_blocks[i])
			regularization = regularization.sum()
		else:
			for i in range(3):
				z = self.fc_dropouts[i](self.fc_blocks[i](z))
		y =  self.pred_y(z)
		return y, regularization

