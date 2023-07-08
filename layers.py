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

'''
https://github.com/yaringal/ConcreteDropout
'''
class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.2, init_max=0.2):
        super(ConcreteDropout, self).__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)
        out = layer(self._concrete_dropout(x, p))
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        input_dimensionality = x[0].numel() # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization
    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = .1
        unif_noise = torch.rand_like(x)
        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
    
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        return x

'''
3D concrete dropout
Adapted from https://github.com/yaringal/ConcreteDropout
'''
class SpatialConcreteDropout(torch.nn.Module):
    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5, 
                    init_min=0.2, init_max=0.2):
        super(SpatialConcreteDropout, self).__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = torch.nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)
        out = layer(self._spatial_concrete_dropout(x, p))
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        input_dimensionality = x[0].numel() # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization
    def _spatial_concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 2/3.
        unif_noise = torch.rand((x.shape[0],1,1,1,1), device=x.device) #3d
        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        return x

'''
3D convolution block
'''
class Conv3d_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, batch_norm=True, max_pool=False, be_num_models=0, be_mixup=False):
        super(Conv3d_Block, self).__init__()
        modules = []
        if be_num_models > 0:
            modules.append(BatchEnsemble_Conv3d(in_channels, out_channels, kernel_size, num_models=be_num_models, mixup=be_mixup))
        else:
            modules.append(nn.Conv3d(in_channels, out_channels, kernel_size))
        if batch_norm:
            modules.append(nn.BatchNorm3d(out_channels))
        modules.append(nn.PReLU())
        if max_pool:
            modules.append(nn.MaxPool3d(2))
        self.conv_block = nn.Sequential(*modules)
    def forward(self, x):
        return self.conv_block(x)

'''
FC convolution block
'''
class FC_Block(nn.Module):
    def __init__(self, in_channels, out_channels, flatten=False, be_num_models=0, be_mixup=False):
        super(FC_Block, self).__init__()
        modules = []
        if flatten:
            modules.append(net_utils.Flatten())
        if be_num_models > 0:
            modules.append(BatchEnsemble_orderFC(in_channels, out_channels, num_models=be_num_models, mixup=be_mixup))
        else:
            modules.append(nn.Linear(in_channels, out_channels))
        modules.append(nn.PReLU())
        self.fc_block = nn.Sequential(*modules)
    def forward(self, x):
        return self.fc_block(x)

'''
Conv3D batch ensemble 
Adapted from https://github.com/giannifranchi/LP_BNN
'''
class BatchEnsemble_Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, first_layer=False, num_models=100, train_gamma=True,
                 bias=True, constant_init=False, p=0.5, random_sign_init=True,
                 mixup=False):
        super(BatchEnsemble_Conv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_channels))
        self.train_gamma = train_gamma
        self.random_sign_init = random_sign_init
        self.constant_init = constant_init
        self.probability = p
        if train_gamma:
            self.gamma = nn.Parameter(torch.Tensor(num_models, out_channels))
        self.num_models = num_models
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = first_layer
        self.mixup = mixup

    def reset_parameters(self):
        if self.constant_init:
            nn.init.constant_(self.alpha, 1.)
            if self.random_sign_init:
                if self.probability  == -1:
                    with torch.no_grad():
                        factor = torch.ones(
                            self.num_models, device=self.alpha.device).bernoulli_(0.5)
                        factor.mul_(2).add_(-1)
                        self.alpha.data = (self.alpha.t() * factor).t()
                        if self.train_gamma:
                            self.gamma.fill_(1.)
                            self.gamma.data = (self.gamma.t() * factor).t()
                elif self.probability  == -2:
                    with torch.no_grad():
                        positives_num = self.num_models // 2
                        factor1 = torch.Tensor([1 for i in range(positives_num)])
                        factor2 = torch.Tensor(
                            [-1 for i in range(self.num_models-positives_num)])
                        factor = torch.cat([factor1, factor2]).to(self.alpha.device)
                        self.alpha.data = (self.alpha.t() * factor).t()
                        if self.train_gamma:
                            self.gamma.fill_(1.)
                            self.gamma.data = (self.gamma.t() * factor).t()
                else:
                    with torch.no_grad():
                        self.alpha.bernoulli_(self.probability)
                        self.alpha.mul_(2).add_(-1)
                        if self.train_gamma:
                            self.gamma.bernoulli_(self.probability)
                            self.gamma.mul_(2).add_(-1)
        else:
            nn.init.normal_(self.alpha, mean=1., std=0.5)
            #nn.init.normal_(self.alpha, mean=1., std=1)
            if self.train_gamma:
                nn.init.normal_(self.gamma, mean=1., std=0.5)
                #nn.init.normal_(self.gamma, mean=1., std=1)
            if self.random_sign_init:
                with torch.no_grad():
                    alpha_coeff = torch.randint_like(self.alpha, low=0, high=2)
                    alpha_coeff.mul_(2).add_(-1)
                    self.alpha *= alpha_coeff
                    if self.train_gamma:
                        gamma_coeff = torch.randint_like(self.gamma, low=0, high=2)
                        gamma_coeff.mul_(2).add_(-1)
                        self.gamma *= gamma_coeff
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def update_indices(self, indices):
        self.indices = indices

    def forward(self, x):
        # if not self.training and self.first_layer:
        #     # Repeated pattern in test: [[A,B,C],[A,B,C]]
        #     x = torch.cat([x for i in range(self.num_models)], dim=0)
        if self.train_gamma:
            num_examples_per_model = int(x.size(0) / self.num_models)
            extra = x.size(0) - (num_examples_per_model * self.num_models)

            if self.mixup:
                # Repeated pattern: [[A,A],[B,B],[C,C]]
                if num_examples_per_model != 0:
                    lam = np.random.beta(0.2, 0.2)
                    i = np.random.randint(self.num_models)
                    j = np.random.randint(self.num_models)
                    alpha = (lam * self.alpha[i] + (1 - lam) * self.alpha[j]).unsqueeze(0)
                    gamma = (lam * self.gamma[i] + (1 - lam) * self.gamma[j]).unsqueeze(0)
                    bias = (lam * self.bias[i] + (1 - lam) * self.bias[j]).unsqueeze(0)
                    for index in range(x.size(0)-1):
                        lam = np.random.beta(0.2, 0.2)
                        i = np.random.randint(self.num_models)
                        j = np.random.randint(self.num_models)
                        next_alpha = (lam * self.alpha[i] + (1 - lam) * self.alpha[j]).unsqueeze(0)
                        alpha = torch.cat([alpha,next_alpha], dim=0)
                        next_gamma = (lam * self.gamma[i] + (1 - lam) * self.gamma[j]).unsqueeze(0)
                        gamma = torch.cat([gamma,next_gamma], dim=0)
                        next_bias = (lam * self.bias[i] + (1 - lam) * self.bias[j]).unsqueeze(0)
                        bias = torch.cat([bias, next_bias], dim=0)
                else:
                    print("Error: TODO")
            else:
                # Repeated pattern: [[A,A],[B,B],[C,C]]
                alpha = torch.cat(
                    [self.alpha for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.in_channels])
                gamma = torch.cat(
                    [self.gamma for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_channels])
                if self.bias is not None:
                    bias = torch.cat(
                        [self.bias for i in range(num_examples_per_model)],
                        dim=1).view([-1, self.out_channels])
                    

            alpha.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)
            gamma.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)
            bias.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)

            if extra != 0:
                alpha = torch.cat([alpha, alpha[:extra]], dim=0)
                gamma = torch.cat([gamma, gamma[:extra]], dim=0)
                if self.bias is not None:
                    bias = torch.cat([bias, bias[:extra]], dim=0)

            result = self.conv(x*alpha)*gamma

            return result + bias if self.bias is not None else result
        else:
            num_examples_per_model = int(x.size(0) / self.num_models)
            # Repeated pattern: [[A,A],[B,B],[C,C]]
            alpha = torch.cat(
                [self.alpha for i in range(num_examples_per_model)],
                dim=1).view([-1, self.in_channels])
            alpha.unsqueeze_(-1).unsqueeze_(-1)

            if self.bias is not None:
                bias = torch.cat(
                    [self.bias for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_channels])
                bias.unsqueeze_(-1).unsqueeze_(-1)
            result = self.conv(x*alpha)
            return result + bias if self.bias is not None else result

# '''
# Fully connected batch ensemble 
# Adapted from https://github.com/giannifranchi/LP_BNN
# '''
# class BatchEnsemble_FC(nn.Module):
#     def __init__(self, in_features, out_features, num_models, \
#                   first_layer=False, bias=True):
#         super(BatchEnsemble_FC, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.fc = nn.Linear(in_features, out_features, bias=False)
#         self.alpha = nn.Parameter(torch.Tensor(num_models, in_features))
#         self.gamma = nn.Parameter(torch.Tensor(num_models, out_features))
#         self.num_models = num_models
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(self.num_models, out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#         self.first_layer = first_layer

#     def reset_parameters(self):
#         nn.init.normal_(self.alpha, mean=1., std=0.1)
#         nn.init.normal_(self.gamma, mean=1., std=0.1)
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)

#     def update_indices(self, indices):
#         self.indices = indices

#     def forward(self, x):
#         # if self.training:
#         curr_bs = x.size(0)
#         makeup_bs = self.num_models - curr_bs
#         if makeup_bs > 0:
#             indices = torch.randint(
#                 high=self.num_models,
#                 size=(curr_bs,), device=self.alpha.device)
#             alpha = torch.index_select(self.alpha, 0, indices)
#             gamma = torch.index_select(self.gamma, 0, indices)
#             bias = torch.index_select(self.bias, 0, indices)
#             result = self.fc(x * alpha) * gamma + bias
#         elif makeup_bs < 0:
#             indices = torch.randint(
#                 high=self.num_models,
#                 size=(curr_bs,), device=self.alpha.device)
#             indices = torch.IntTensor([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]).to(self.alpha.device)
#             alpha = torch.index_select(self.alpha, 0, indices)
#             gamma = torch.index_select(self.gamma, 0, indices)
#             bias = torch.index_select(self.bias, 0, indices)
#             result = self.fc(x * alpha) * gamma + bias
#         else:
#             result = self.fc(x * self.alpha) * self.gamma + self.bias
#         return result[:curr_bs]
#         # else:
#         #     # print("Here")
#         #     # if self.first_layer:
#         #     #     # Repeated pattern: [[A,B,C],[A,B,C]]
#         #     #     x = torch.cat([x for i in range(self.num_models)], dim=0)
#         #     # Repeated pattern: [[A,A],[B,B],[C,C]]
#         #     batch_size = int(x.size(0) / self.num_models)
#         #     alpha = torch.cat(
#         #         [self.alpha for i in range(batch_size)],
#         #         dim=1).view([-1, self.in_features])
#         #     gamma = torch.cat(
#         #         [self.gamma for i in range(batch_size)],
#         #         dim=1).view([-1, self.out_features])
#         #     bias = torch.cat(
#         #         [self.bias for i in range(batch_size)],
#         #         dim=1).view([-1, self.out_features])
#         #     result = self.fc(x * alpha) * gamma + bias
#         #     return result

class BatchEnsemble_orderFC(nn.Module):
    def __init__(self, in_features, out_features, num_models, first_layer=False,
                 bias=True, constant_init=False, p=0.5, random_sign_init=True,
                 mixup=True):
        super(BatchEnsemble_orderFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_features))
        self.gamma = nn.Parameter(torch.Tensor(num_models, out_features))
        #self.alpha = torch.Tensor(num_models, in_features).cuda()
        #self.gamma = torch.Tensor(num_models, out_features).cuda()
        self.num_models = num_models
        self.random_sign_init = random_sign_init
        self.constant_init = constant_init
        self.probability = p
        if bias:
            #self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = first_layer
        self.mixup = mixup

    def reset_parameters(self):
        if self.constant_init:
            nn.init.constant_(self.alpha, 1.)
            nn.init.constant_(self.gamma, 1.)
            if self.random_sign_init:
                if self.probability  == -1:
                    with torch.no_grad():
                        factor = torch.ones(
                            self.num_models, device=self.alpha.device).bernoulli_(0.5)
                        factor.mul_(2).add_(-1)
                        self.alpha.data = (self.alpha.t() * factor).t()
                        self.gamma.data = (self.gamma.t() * factor).t()
                elif self.probability  == -2:
                    with torch.no_grad():
                        positives_num = self.num_models // 2
                        factor1 = torch.Tensor([1 for i in range(positives_num)])
                        factor2 = torch.Tensor(
                            [-1 for i in range(self.num_models-positives_num)])
                        factor = torch.cat([factor1, factor2]).to(self.alpha.device)
                        self.alpha.data = (self.alpha.t() * factor).t()
                        self.gamma.data = (self.gamma.t() * factor).t()
                else:
                    with torch.no_grad():
                        self.alpha.bernoulli_(self.probability)
                        self.alpha.mul_(2).add_(-1)
                        self.gamma.bernoulli_(self.probability)
                        self.gamma.mul_(2).add_(-1)
        else:
            nn.init.normal_(self.alpha, mean=1., std=0.5)
            nn.init.normal_(self.gamma, mean=1., std=0.5)
            #nn.init.normal_(self.alpha, mean=1., std=1)
            #nn.init.normal_(self.gamma, mean=1., std=1)
            if self.random_sign_init:
                with torch.no_grad():
                    alpha_coeff = torch.randint_like(self.alpha, low=0, high=2)
                    alpha_coeff.mul_(2).add_(-1)
                    self.alpha *= alpha_coeff
                    gamma_coeff = torch.randint_like(self.gamma, low=0, high=2)
                    gamma_coeff.mul_(2).add_(-1)
                    self.gamma *= gamma_coeff
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def update_indices(self, indices):
        self.indices = indices

    def forward(self, x):
        # if not self.training and self.first_layer:
        #     # Repeated pattern in test: [[A,B,C],[A,B,C]]
        #     x = torch.cat([x for i in range(self.num_models)], dim=0)
        num_examples_per_model = int(x.size(0) / self.num_models)
        extra = x.size(0) - (num_examples_per_model * self.num_models)
        if self.mixup: 
            # Repeated pattern: [[A,A],[B,B],[C,C]]
            if num_examples_per_model != 0:
                lam = np.random.beta(0.2, 0.2)
                i = np.random.randint(self.num_models)
                j = np.random.randint(self.num_models)
                alpha = (lam * self.alpha[i] + (1 - lam) * self.alpha[j]).unsqueeze(0)
                gamma = (lam * self.gamma[i] + (1 - lam) * self.gamma[j]).unsqueeze(0)
                bias = (lam * self.bias[i] + (1 - lam) * self.bias[j]).unsqueeze(0)
                for index in range(x.size(0)-1):
                    lam = np.random.beta(0.2, 0.2)
                    i = np.random.randint(self.num_models)
                    j = np.random.randint(self.num_models)
                    next_alpha = (lam * self.alpha[i] + (1 - lam) * self.alpha[j]).unsqueeze(0)
                    alpha = torch.cat([alpha,next_alpha], dim=0)
                    next_gamma = (lam * self.gamma[i] + (1 - lam) * self.gamma[j]).unsqueeze(0)
                    gamma = torch.cat([gamma,next_gamma], dim=0)
                    next_bias = (lam * self.bias[i] + (1 - lam) * self.bias[j]).unsqueeze(0)
                    bias = torch.cat([bias, next_bias], dim=0)
            else:
                print("Error: TODO")
        else:
            # Repeated pattern: [[A,A],[B,B],[C,C]]
            if num_examples_per_model != 0:
                alpha = torch.cat(
                    [self.alpha for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.in_features])
                gamma = torch.cat(
                    [self.gamma for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_features])
                if self.bias is not None:
                    bias = torch.cat(
                        [self.bias for i in range(num_examples_per_model)],
                        dim=1).view([-1, self.out_features])
            else:
                alpha = self.alpha.clone()
                gamma = self.gamma.clone()
                if self.bias is not None:
                    bias = self.bias.clone()
        if extra != 0:
            alpha = torch.cat([alpha, alpha[:extra]], dim=0)
            gamma = torch.cat([gamma, gamma[:extra]], dim=0)
            bias = torch.cat([bias, bias[:extra]], dim=0)
        result = self.fc(x*alpha)*gamma
        return result + bias if self.bias is not None else result
