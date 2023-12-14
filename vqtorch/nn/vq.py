import torch
import torch.nn as nn
import torch.nn.functional as F

from vqtorch.dists import get_dist_fns
import vqtorch
from vqtorch.norms import with_codebook_normalization
from .vq_base import _VQBaseLayer
from .affine import AffineTransform


class VectorQuant(_VQBaseLayer):
	"""
	Vector quantization layer using straight-through estimation.

	Args:
		feature_size (int): feature dimension corresponding to the vectors
		num_codes (int): number of vectors in the codebook
		beta (float): commitment loss weighting
		sync_nu (float): sync loss weighting
		affine_lr (float): learning rate for affine transform
		affine_groups (int): number of affine parameter groups
		replace_freq (int): frequency to replace dead codes
		inplace_optimizer (Optimizer): optimizer for inplace codebook updates
		**kwargs: additional arguments for _VQBaseLayer
	
	Returns:
		Quantized vector z_q and return dict
	"""


	def __init__(
			self,
			feature_size : int,
			num_codes : int,
			beta : float = 0.97125 , # 0.9475 8.418
			sync_nu : float = 0.0,
			affine_lr:	float = 0.0,
			affine_groups: int = 1,
			replace_freq: int = 0,
			inplace_optimizer: torch.optim.Optimizer = None,
			using_statistics: bool = False,
			use_learnable_std: bool = False,
			use_learnable_mean: bool = False,
			alter_penalty : bool = False,
			**kwargs,
			):

		super().__init__(feature_size, num_codes, **kwargs)
		self.loss_fn, self.dist_fn = get_dist_fns('euclidean')
		self.alter_penalty = alter_penalty

		if beta < 0.0 or beta > 1.0:
			raise ValueError(f'beta must be in [0, 1] but got {beta}')
			
		self.beta = beta
		self.nu = sync_nu
		self.affine_lr = affine_lr
		self.codebook = nn.Embedding(self.num_codes, self.feature_size)

		if inplace_optimizer is not None:
			if beta != 1.0:
				raise ValueError('inplace_optimizer can only be used with beta=1.0')
			self.inplace_codebook_optimizer = inplace_optimizer(self.codebook.parameters())

		if affine_lr > 0 or use_learnable_mean or use_learnable_std:
			# defaults to using learnable affine parameters
			self.affine_transform = AffineTransform(
										self.code_vector_size,
										use_running_statistics=using_statistics,
										use_learnable_std=use_learnable_std,
										use_learnable_mean=use_learnable_mean,
										lr_scale=affine_lr,
										num_groups=affine_groups,
										)
		if replace_freq > 0:
			vqtorch.nn.utils.lru_replacement(self, rho=0.01, timeout=replace_freq)
		return


	def straight_through_approximation(self, z, z_q):
		""" passed gradient from z_q to z """
		if self.nu > 0:
			z_q = z + (z_q - z).detach() + (self.nu * z_q) + (-self.nu * z_q).detach()
		else:
			z_q = z + (z_q - z).detach()
		return z_q


	def compute_loss(self, z_e, z_q):
		""" computes loss between z and z_q """
		return ((1.0 - self.beta) * self.loss_fn(z_e, z_q.detach()) + \
					  (self.beta) * self.loss_fn(z_e.detach(), z_q))


	def quantize(self, codebook, z):
		"""
		Quantizes the latent codes z with the codebook

		Args:
			codebook (Tensor): B x F
			z (Tensor): B x ... x F
		"""

		# reshape to (BHWG x F//G) and compute distance
		z_shape = z.shape[:-1]
		z_flat = z.view(z.size(0), -1, z.size(-1))

		if hasattr(self, 'affine_transform'):
			self.affine_transform.update_running_statistics(z_flat, codebook)
			codebook, alpha = self.affine_transform(codebook)

		with torch.no_grad():
			dist_out = self.dist_fn(
							tensor=z_flat,
							codebook=codebook,
							topk=self.topk,
							compute_chunk_size=self.cdist_chunk_size,
							half_precision=(z.is_cuda),
							)

			d = dist_out['d'].view(z_shape)
			q = dist_out['q'].view(z_shape).long()

		z_q = F.embedding(q, codebook)

		if self.training and hasattr(self, 'inplace_codebook_optimizer'):
			# update codebook inplace 
			((z_q - z.detach()) ** 2).mean().backward()
			self.inplace_codebook_optimizer.step()
			self.inplace_codebook_optimizer.zero_grad()

			# forward pass again with the update codebook
			z_q = F.embedding(q, codebook)

			# NOTE to save compute, we assumed Q did not change.
		return z_q, d, q

	def alpha_loss(self):
		if hasattr(self, 'affine_transform'):
			return self.affine_transform.alpha_loss_2() if self.alter_penalty else self.affine_transform.alpha_loss_1()
		return 1.0

	@torch.no_grad()
	def get_codebook(self):
		cb = self.codebook.weight
		# if hasattr(self, 'affine_transform'):
		# 	cb = self.affine_transform(cb)
		return cb

	@torch.no_grad()
	def get_alpha(self):
		if hasattr(self, 'affine_transform'):
			return self.affine_transform.get_alpha()

	def get_codebook_affine_params(self):
		if hasattr(self, 'affine_transform'):
			return self.affine_transform.get_affine_params()
		return None

	@with_codebook_normalization
	def forward(self, z):

		######
		## (1) formatting data by groups and invariant to dim
		######

		z = self.prepare_inputs(z, self.groups)

		if not self.enabled:
			z = self.to_original_format(z)
			return z, {}

		######
		## (2) quantize latent vector
		######

		z_q, d, q = self.quantize(self.codebook.weight, z)

		e_mean = F.one_hot(q, num_classes=self.num_codes).view(-1, self.num_codes).float().mean(0)
		# perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
		perplexity = 2 ** (-torch.sum(e_mean * torch.log2(e_mean + 1e-10)))
		active_ratio = q.unique().numel() / self.num_codes * 100

		to_return = {
			'z'  : z,               # each group input z_e
			'z_q': z_q,             # quantized output z_q
			'd'  : d,               # distance function for each group
			'q'	 : q,				# codes
			'loss': self.compute_loss(z, z_q).mean(),
			'perplexity': perplexity,
			'active_ratio': active_ratio,
			}

		z_q = self.straight_through_approximation(z, z_q)
		z_q = self.to_original_format(z_q)

		return z_q, to_return


def train(model, train_loader, train_iterations=1000, alpha=10):
	def iterate_dataset(data_loader):
		data_iter = iter(data_loader)
		while True:
			try:
				x, y = next(data_iter)
			except StopIteration:
				data_iter = iter(data_loader)
				x, y = next(data_iter)
			yield x.cuda(), y.cuda()

	for _ in (pbar := trange(train_iterations)):
		opt.zero_grad()
		x, _ = next(iterate_dataset(train_loader))
		out, vq_out = model(x)
		rec_loss = (out - x).abs().mean()
		cmt_loss = vq_out['loss']
		(rec_loss + alpha * cmt_loss).backward()

		opt.step()
		pbar.set_description(f'rec loss: {rec_loss.item():.3f} | ' + \
							 f'cmt loss: {cmt_loss.item():.3f} | ' + \
							 f'active %: {vq_out["q"].unique().numel() / num_codes * 100:.3f}')
	return