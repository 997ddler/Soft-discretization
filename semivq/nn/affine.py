import math
import time
import warnings
import numpy as np
from semivq.nn.mlp import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineTransform(nn.Module):
	def __init__(
			self, 
			feature_size, 
			use_running_statistics=False, 
			momentum=0.1,
			use_learnable_std=False,
			lr_scale=1,
			num_groups=1,

			):
		super().__init__()

		self.use_running_statistics = use_running_statistics
		self.num_groups = num_groups

		self.use_learnable_std = use_learnable_std
		self.arr_alpha0 = []
		self.arr_alpha1 = []
		self.arr_alpha_mean = []
		self.alpha = None
		self.mean = None
		self.iter = 0
		if use_running_statistics or use_learnable_std:
			self.momentum = momentum
			self.register_buffer('running_statistics_initialized', torch.zeros(1))
			self.register_buffer('running_ze_mean', torch.zeros(num_groups, feature_size))
			self.register_buffer('running_ze_var', torch.ones(num_groups, feature_size))

			self.register_buffer('running_c_mean', torch.zeros(num_groups, feature_size))
			self.register_buffer('running_c_var', torch.ones(num_groups, feature_size))
			if use_learnable_std:
				self.mlp_std = MLP(1026, 1, 256)
		else:
			self.scale = nn.parameter.Parameter(torch.zeros(num_groups, feature_size))
			self.bias = nn.parameter.Parameter(torch.zeros(num_groups, feature_size))
			self.lr_scale = lr_scale
		return

	@torch.no_grad()
	def update_running_statistics(self, z_e, c):
		# we find it helpful to often to make an under-estimation on the
		# z_e embedding statistics. Empirically we observe a slight
		# over-estimation of the statistics, causing the straight-through
		# estimation to grow indefinitely. While this is not an issue
		# for most model architecture, some model architectures that don't
		# have normalized bottlenecks, can cause it to eventually explode.
        # placing the VQ layer in certain layers of ViT exhibits this behavior

		if self.training and (self.use_running_statistics or self.use_learnable_std):
			unbiased = False

			ze_mean = z_e.mean([0, 1]).unsqueeze(0)
			ze_var = z_e.var([0, 1], unbiased=unbiased).unsqueeze(0)

			c_mean = c.mean([0]).unsqueeze(0)
			c_var = c.var([0], unbiased=unbiased).unsqueeze(0)

			if not self.running_statistics_initialized:
				self.running_ze_mean.data.copy_(ze_mean)
				self.running_ze_var.data.copy_(ze_var)
				self.running_c_mean.data.copy_(c_mean)
				self.running_c_var.data.copy_(c_var)
				self.running_statistics_initialized.fill_(1)
			else:
				self.running_ze_mean = (self.momentum * ze_mean) + (1 - self.momentum) * self.running_ze_mean
				self.running_ze_var = (self.momentum * ze_var) + (1 - self.momentum) * self.running_ze_var
				self.running_c_mean = (self.momentum * c_mean) + (1 - self.momentum) * self.running_c_mean
				self.running_c_var = (self.momentum * c_var) + (1 - self.momentum) * self.running_c_var

		# wd = 0.9998 # 0.995
		# self.running_ze_mean = wd * self.running_ze_mean
		# self.running_ze_var = wd * self.running_ze_var
		return

	def get_inner_layer(self):
		if self.use_learnable_std:
			return self.mlp_std
		else:
			raise Exception('inner layer is None')


	def alpha_loss_1(self):
		if self.use_learnable_std:
			return - (self.alpha ** 2).sum()
		else:
			return 1.0

	def alpha_loss_2(self):
		if self.use_learnable_std:
			return ((1 - self.alpha) ** 2).sum()
		else:
			return 1.0

	def alpha_loss_3(self):
		if self.use_learnable_std:
			self.iter = self.iter + 1
			return -self.iter * 0.5 * (self.alpha ** 2).sum()
		else:
			return 1.0

	def get_dynamic_info(self):
		if self.use_learnable_std:
			return self.arr_alpha0, self.arr_alpha1, self.arr_alpha_mean
		return None

	def forward(self, codebook):
		scale0, bias = self.get_affine_params()
		self.mean = bias
		if self.use_learnable_std:
			input = torch.cat([codebook, torch.squeeze(scale0, 1), torch.squeeze(bias, 1)], dim=0).T
			#input = codebook.T
			scale = self.mlp_std(input).T.unsqueeze(1)
			self.scale = scale.data
			self.alpha = scale.data
		else:
			scale = scale0
			self.alpha = scale.data
		self.arr_alpha0.append(scale.data[0][0][0].detach().cpu().numpy())
		self.arr_alpha1.append(scale.data[0][0][1].detach().cpu().numpy())
		self.arr_alpha_mean.append(torch.mean(scale.data[0][0]).detach().cpu().numpy())
		n, c = codebook.shape
		codebook = codebook.view(self.num_groups, -1, codebook.shape[-1])
		codebook = bias + scale * codebook
		return codebook.reshape(n, c), scale.data

	def get_affine_params(self):
		if self.use_running_statistics or self.use_learnable_std:
			scale = (self.running_ze_var / (self.running_c_var + 1e-8)).sqrt()
			bias = - scale * self.running_c_mean + self.running_ze_mean
		else:
			scale = (1. + self.lr_scale * self.scale)
			bias = self.lr_scale * self.bias
		return scale.unsqueeze(1), bias.unsqueeze(1)

	def get_last_mean(self):
		return self.mean

	def get_alpha(self):
		return self.alpha
