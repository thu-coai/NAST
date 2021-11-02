import torch
import torch.nn.functional as F
import numpy as np

from .cuda_helper import cuda, Tensor

def compile(input_dim, output_dim):
	swap_command = []
	expand_command = []
	input_dim = list(input_dim)

	for i, c in enumerate(output_dim):
		if c == "_":
			expand_command.append(i)
		else:
			oldid = input_dim.index(c)
			swap_command.append(oldid)

	return swap_command, expand_command

RESHAPE_CACHE = {}

def reshape(x, input_dim, output_dim, *args):
	global RESHAPE_CACHE
	assert len(x.shape) == len(input_dim)

	command = RESHAPE_CACHE.get((input_dim, output_dim), None)
	if command is None:
		command = RESHAPE_CACHE[(input_dim, output_dim)] = compile(input_dim, output_dim)

	swap_command, expand_command = command
	x = x.permute(*swap_command)
	expand_size = [-1 for c in output_dim]
	for i, pos in enumerate(expand_command):
		x = x.unsqueeze(pos)
		expand_size[pos] = args[i]

	x = x.expand(*expand_size)
	return x

def unsqueeze(x, li):
	for s in li:
		x = x.unsqueeze(s)
	return x

def broadcast(x, y, ignore=None):
	assert len(x.shape) == len(y.shape)
	shape = []
	ignore = ignore or []
	for i, (a, b) in enumerate(zip(x.shape, y.shape)):
		if i in ignore or (i - len(x.shape)) in ignore:
			shape.append(-1)
		elif a == 1:
			shape.append(b)
		else:
			shape.append(a)
	return x.expand(*shape), y.expand(*shape)

def cdist2(x, y, eps=1e-12):
    # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>
    x_sq_norm = x.pow(2).sum(dim=-1)
    y_sq_norm = y.pow(2).sum(dim=-1)
    x_dot_y = torch.einsum("ik,jk->ij", x, y)
    sq_dist = x_sq_norm.unsqueeze(dim=1) + y_sq_norm.unsqueeze(dim=0) - 2*x_dot_y
    # For numerical issues
    sq_dist.clamp_(min=eps)
    return torch.sqrt(sq_dist)

def cdist_nobatch(x, y):
	# x = a * b * ... * d_emb
	# y = c * d * ... * d_emb

	d_emb = x.shape[-1]
	x_flatten = x.reshape(-1, d_emb)
	y_flatten = y.reshape(-1, d_emb)

	dis = cdist2(x_flatten, y_flatten)

	return dis.reshape(*(x.shape[:-1] + y.shape[:-1]))



def sequence_pooling(x, sent_length, pool="avg"):
	mask = generateMask(x.shape[1], sent_length).transpose(0, 1)
	for i in range(len(x.shape) - 2):
		mask = mask.unsqueeze(-1)

	if pool == "avg":
		return (mask * x).sum(dim=1) / mask.sum(1)
	elif pool == "sum":
		return (mask * x).sum(dim=1)
	elif pool == "min":
		x = x * mask + (1 - mask) * torch.ones_like(mask) * 1e8
		return x.min(dim=1)[0]
	elif pool == "max":
		x = x * mask - (1 - mask) * torch.ones_like(mask) * 1e8
		return x.max(dim=1)[0]

def generateMask(seqlen, length, type=int, device=None):
	return Tensor(
		(np.expand_dims(np.arange(seqlen), 1) < np.expand_dims(length, 0)).astype(type), device=device)

def tensor(x, device=None, dlike=None):
	if dlike is not None:
		return torch.tensor(x, dtype=torch.float, device=dlike.device)
	else:
		return torch.tensor(x, dtype=torch.float, device=device)

def longtensor(x, device=None, dlike=None):
	if dlike is not None:
		return torch.tensor(x, dtype=torch.long, device=dlike.device)
	else:
		return torch.tensor(x, dtype=torch.long, device=device)

def zerot(*size, dtype=None, device=None, dlike=None):
	if dlike is not None:
		return torch.zeros(*size, dtype=dlike.dtype, device=dlike.device)
	else:
		return torch.zeros(*size, dtype=dtype, device=device)

def onet(*size, dtype=None, device=None, dlike=None):
	if dlike is not None:
		return torch.ones(*size, dtype=dlike.dtype, device=dlike.device)
	else:
		return torch.ones(*size, dtype=dtype, device=device)
