# coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

from .cuda_helper import Tensor, cuda, ones, zeros

def gumbel_softmax(logits, tau=1, dim=-1):
	# type: (Tensor, float, bool, float, int) -> Tensor
	r"""
	Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

	Args:
	  logits: `[..., num_features]` unnormalized log probabilities
	  tau: non-negative scalar temperature
	  hard: if ``True``, the returned samples will be discretized as one-hot vectors,
			but will be differentiated as if it is the soft sample in autograd
	  dim (int): A dimension along which softmax will be computed. Default: -1.

	Returns:
	  Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
	  If ``hard=True``, the returned samples will be one-hot, otherwise they will
	  be probability distributions that sum to 1 across `dim`.
	"""

	# workaround for bug in torch 1.1.0
	gumbels = torch.min(-torch.empty_like(logits).exponential_().log(), ones(1, device=logits) * 1e10)  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
	y_soft = gumbels.softmax(dim)

	ret = y_soft
	return ret

def gumbel_max(logits, tau=1, dim=-1):
	gumbels = torch.min(-torch.empty_like(logits).exponential_().log(), ones(1, device=logits) * 1e10)  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
	y_soft = gumbels.softmax(dim)

	# Straight through.
	index = y_soft.max(dim, keepdim=True)[1]
	y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
	ret = y_hard - y_soft.detach() + y_soft
	return ret, y_soft

def straight_max(logits, dim=-1, allow_gradient=False):
	y_soft = logits.softmax(dim)
	# Straight through.
	index = y_soft.max(dim, keepdim=True)[1]
	y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
	if allow_gradient:
		y_hard = y_hard - y_soft.detach() + y_soft
	return y_hard, y_soft

def onehot(sent, vocab_size):
	y_hard = torch.zeros(*(sent.shape + (vocab_size,)), device=sent.device).scatter_(-1, sent.unsqueeze(-1), 1.0)
	return y_hard

def gumbel_max_with_mask(logits, mask, tau=1, dim=-1):
	gumbels = torch.min(-torch.empty_like(logits).exponential_().log(), ones(1, device=logits) * 1e10)  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
	gumbels = gumbels.masked_fill(mask==0, -1e9)
	y_soft = gumbels.softmax(dim)

	# Straight through.
	index = y_soft.max(dim, keepdim=True)[1]
	y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
	ret = y_hard - y_soft.detach() + y_soft
	return ret

# no cdist in torch??
def cdist(x1, x2, p=2):
	r"""Computes batched the p-norm distance between each pair of the two collections of row vectors.
	Args:
		x1 (Tensor): input tensor of shape :math:`B \times P \times M`.
		x2 (Tensor): input tensor of shape :math:`B \times R \times M`.
		p: p value for the p-norm distance to calculate between each vector pair
			:math:`\in [0, \infty]`.
		compute_mode:
			'use_mm_for_euclid_dist_if_necessary' - will use matrix multiplication approach to calculate
			euclidean distance (p = 2) if P > 25 or R > 25
			'use_mm_for_euclid_dist' - will always use matrix multiplication approach to calculate
			euclidean distance (p = 2)
			'donot_use_mm_for_euclid_dist' - will never use matrix multiplication approach to calculate
			euclidean distance (p = 2)
			Default: use_mm_for_euclid_dist_if_necessary.
	If x1 has shape :math:`B \times P \times M` and x2 has shape :math:`B \times R \times M` then the
	output will have shape :math:`B \times P \times R`.
	This function is equivalent to `scipy.spatial.distance.cdist(input,'minkowski', p=p)`
	if :math:`p \in (0, \infty)`. When :math:`p = 0` it is equivalent to
	`scipy.spatial.distance.cdist(input, 'hamming') * M`. When :math:`p = \infty`, the closest
	scipy function is `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`.
	Example:
		>>> a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
		>>> a
		tensor([[ 0.9041,  0.0196],
				[-0.3108, -2.4423],
				[-0.4821,  1.0590]])
		>>> b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
		>>> b
		tensor([[-2.1763, -0.4713],
				[-0.6986,  1.3702]])
		>>> torch.cdist(a, b, p=2)
		tensor([[3.1193, 2.0959],
				[2.7138, 3.8322],
				[2.2830, 0.3791]])
	"""
	return torch._C._VariableFunctions.cdist(x1, x2, p)

# pylint: disable=W0221
class RecenterGradient(Function):
	@staticmethod
	def forward(ctx, one_hot, emb, dim=-1):
		assert dim == -1
		_, idx = torch.max(one_hot, dim=-1)
		out = F.embedding(idx, emb)
		ctx.save_for_backward(one_hot, idx, emb, out)
		return out

	@staticmethod
	def backward(ctx, grad_output):
		one_hot, idx, emb, out = ctx.saved_variables
		# onehot: idx[:-1], emb.shape[0]
		grad_one_hot, grad_emb = None, None
		if ctx.needs_input_grad[0]:
		#emb emb.shape[0], emb.shape[1]
			out_flatten = out.reshape(1, -1, emb.shape[1])
			dist = cdist(out_flatten, emb.unsqueeze(0)).reshape(*(out.shape[:-1] + emb.shape[:1]))
			A = 1 / (dist ** 2 + 1).sqrt()
			C = torch.einsum("ij,...j->...i", emb, grad_output)
			D = torch.einsum("...j,...j->...", out, grad_output)
			B = C - D.unsqueeze(-1)
			grad_one_hot = A * B
		if ctx.needs_input_grad[1]:
			grad_emb = torch.einsum("abj,abi->ji", one_hot, grad_output)
		return grad_one_hot, grad_emb
recenter_gradient = RecenterGradient.apply

class RebarGradientFunction(Function):
	@staticmethod
	def forward(ctx, logits, emb, f_mean, g_mean, g2_mean, fg_mean, tau=1, n_iters=3):
		gumbels = torch.min(-torch.empty_like(logits).exponential_().log(), ones(1, device=logits) * 1e10)  # ~Gumbel(0,1)

		with torch.enable_grad():
			logits.requires_grad_()
			gumbels_unnorm = logits + gumbels
			gumbels = gumbels_unnorm / tau  # ~Gumbel(logits,tau)
			logp = gumbels.log_softmax(dim=-1)
			logp_max, idx = logp.max(dim=-1)
			gumbels1_softmax = logp.exp()

		f = F.embedding(idx, emb)
		mask = zeros(*logits.shape, device=logits).scatter_(-1, idx.unsqueeze(-1), 1)
		target_logits = logits.masked_fill(mask == 0, -1e9).max(dim=-1, keepdim=True)[0]

		gumbels_target = torch.min(-torch.empty_like(logits[:, :, :1]).exponential_().log(), ones(1, device=logits) * 1e10)  # ~Gumbel(0,1)
		for _ in range(n_iters):
			upper = gumbels_target + target_logits - logits
			upper = (-(-upper).exp()).exp()
			gumbels_other = torch.empty_like(logits).uniform_() * upper
			gumbels_other = -(-(gumbels_other + 1e-20).log()+1e-20).log()

			lower = (gumbels_other + logits - target_logits).masked_fill(mask == 1, -1e9).max(dim=-1, keepdim=True)[0]
			lower = (-(-lower).exp()).exp()
			gumbels_target = torch.empty_like(logits[:, :, :1]).uniform_() * (1 - lower) + lower
			gumbels_target = -(-(gumbels_target + 1e-20).log()+1e-20).log()
		gumbels2_unnorm = gumbels_other * (1 - mask) + gumbels_target * mask

		with torch.enable_grad():
			gumbels2_unnorm = logits + gumbels2_unnorm
			gumbels2 = gumbels2_unnorm / tau
			gumbels2_softmax = gumbels2.softmax(dim=-1)

		#assert (gumbels2_unnorm.max(dim=-1)[1] == idx).min() == 1

		g = torch.einsum("ij,...i->...j", emb, gumbels2_softmax)

		eta = ((fg_mean - f_mean * g_mean) / (g2_mean - g_mean ** 2)).unsqueeze(0).unsqueeze(0)

		ctx.save_for_backward(logits, emb, f, eta, g, logp_max, gumbels1_softmax, gumbels2_softmax)
		return f, g, gumbels1_softmax.detach()

	@staticmethod
	def backward(ctx, grad_output, g_grad, gumbel_grad):
		#assert g_grad is None
		#assert gumbel_grad is None
		logits, emb, f, eta, g, logp_max, gumbels1_softmax, gumbels2_softmax = ctx.saved_variables

		with torch.enable_grad():
			res = (f - eta * g).detach() * logp_max.unsqueeze(-1) + eta.detach() * torch.einsum("ij,...i->...j", emb.detach(), gumbels1_softmax - gumbels2_softmax)
			logits_grad = torch.autograd.grad(res, logits, grad_output)[0]
		return logits_grad, None, None, None, None, None, None, None

class RebarGradient(nn.Module):
	def __init__(self, num_features, n_iters=3, tau=1, betas=(0.99, 0.999)):
		super().__init__()

		self.n_iters = n_iters
		self.tau = tau
		self.betas = betas

		self.register_buffer('f_mean', torch.zeros(num_features))
		self.register_buffer('g_mean', torch.zeros(num_features))
		self.register_buffer('g2_mean', torch.ones(num_features))
		self.register_buffer('fg_mean', torch.ones(num_features))

	def forward(self, logits, emb, sent_mask, mode):
		if mode == "gumbel":
			gumbel = gumbel_softmax(logits, tau=self.tau, dim=-1)
			_, idx = gumbel.max(dim=-1)
			f = F.embedding(idx, emb)
		elif mode == "rebar":
			f, g, gumbel = RebarGradientFunction.apply(logits, emb, self.f_mean, self.g_mean, self.g2_mean, self.fg_mean, self.tau, self.n_iters)
			self.f_mean.mul_(self.betas[0]).add_(1 - self.betas[0], (f.detach() * sent_mask.unsqueeze(-1)).sum(0).sum(0) / sent_mask.sum())
			self.g_mean.mul_(self.betas[0]).add_(1 - self.betas[0], (g.detach() * sent_mask.unsqueeze(-1)).sum(0).sum(0) / sent_mask.sum())
			self.g2_mean.mul_(self.betas[1]).add_(1 - self.betas[1], (g.detach() ** 2 * sent_mask.unsqueeze(-1)).sum(0).sum(0) / sent_mask.sum())
			self.fg_mean.mul_(self.betas[1]).add_(1 - self.betas[1], (f.detach() * g.detach() * sent_mask.unsqueeze(-1)).sum(0).sum(0) / sent_mask.sum())
			_, idx = gumbel.max(dim=-1)
		elif mode == "max":
			gumbel = logits.softmax(dim=-1)
			_, idx = gumbel.max(dim=-1)
			f = F.embedding(idx, emb)

		hard_gumbel = torch.zeros_like(logits).scatter_(-1, idx.unsqueeze(dim=-1), 1.0)
		hard_gumbel = hard_gumbel - gumbel.detach() + gumbel

		return f, gumbel, hard_gumbel

# def gumbel_softmax(inp, alpha, beta):
# 	g = Tensor(inp.size()).uniform_(0.0001, 0.9999)
# 	g = Variable(-torch.log(-torch.log(g)))
# 	inp_g = F.softmax((F.log_softmax(inp, dim=-1) + g * alpha) * beta, dim=-1)
# 	return inp_g

# def gumbel_max(inp, alpha, beta):
# 	g = Tensor(inp.size()).uniform_(0.0001, 0.9999)
# 	g = Variable(-torch.log(-torch.log(g)))
# 	inp_g = F.softmax((F.log_softmax(inp, dim=1) + g * alpha) * beta, dim=1)
# 	shape = inp_g.shape
# 	output, idx = StraightThrough.apply(inp_g.reshape(-1, shape[-1]))
# 	return output.reshape(*shape), idx.reshape(*(list(shape[:-1]) + [-1]))

# def gumbel_max_binary(inp, alpha, beta):
# 	inp = torch.cat([1-inp, inp], 1)
# 	g = Tensor(inp.size()).uniform_(0.0001, 0.9999)
# 	g = Variable(-torch.log(-torch.log(g)))
# 	inp_g = F.softmax((torch.log(inp) + g * alpha) * beta, dim=1)
# 	return StraightThrough.apply(inp_g)

# # pylint: disable=W0221
# class StraightThrough(Function):
# 	@staticmethod
# 	def forward(ctx, inp):
# 		#ctx.save_for_backward(inp)
# 		_, idx = torch.max(inp, dim=1)
# 		output = Tensor(inp.size()).zero_()
# 		output[range(inp.size()[0]), idx] = 1
# 		return output, idx

# 	@staticmethod
# 	def backward(ctx, grad_output, _):
# 		#inp = ctx.saved_variables
# 		return grad_output.clone()
