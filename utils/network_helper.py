# coding:utf-8
import logging
from collections import OrderedDict
import re

from torch import nn
from torch.nn import Parameter
from .module_helper import BaseModule

# pylint: disable=W0223, W0221
class BaseNetwork(BaseModule):
	def __init__(self, param, collection_name=None):
		super().__init__()
		args = param.args
		volatile = param.volatile
		self.args = args
		self.volatile = volatile
		self.param = param
		self.collection_name = collection_name or []
		self.collection_name.append("all")

	def remove_collection_name(self, name):
		for cn in self.collection_name:
			name = re.sub(r'_' + cn + r'(?![a-zA-Z])', '', name)
			name = re.sub(r'_exclude_' + cn + r'(?![a-zA-Z])', '', name)
		return name

	def in_collection(self, name, _set):
		for cn in _set:
			if re.search(r"_" + cn + r'(?![a-zA-Z])', name) and not re.search(r"_exclude_" + cn + r'(?![a-zA-Z])', name):
				return True
		return False

	def get_parameters_by_name(self, req=None, silent=False):
		if not silent:
			logging.info("parameters with name: %s", req)
		for name, param in self.named_parameters():
			if req is None:
				if not re.search(r"exclude_all(?![a-zA-Z])'", name):
					if not silent:
						logging.info("\t%s", name)
					yield param
			elif re.search(r"_" + req + r'(?![a-zA-Z])', name) and not re.search(r"_exclude_" + req + r'(?![a-zA-Z])', name) \
					and not re.search(r"_exclude_all(?![a-zA-Z])", name):
				if not silent:
					logging.info("\t%s", name)
				yield param

	def save_gradient_by_name(self, req=None):
		res = {}
		for i, param in enumerate(self.get_parameters_by_name(req, True)):
			if param.grad is not None:
				res[i] = param.grad.detach()
		return res

	def update_gradient_by_name(self, dic, req=None):
		for i, param in enumerate(self.get_parameters_by_name(req, True)):
			if i not in dic:
				continue
			if param.grad is None:
				param.grad = dic[i]
			else:
				param.grad += dic[i]

	def load_state_dict(self, state_dict, exclude_set=None, whitelist=None):
		"""Copies parameters and buffers from :attr:`state_dict` into
		this module and its descendants. The keys of :attr:`state_dict` must
		exactly match the keys returned by this module's :func:`state_dict()`
		function.

		Arguments:
			state_dict (dict): A dict containing parameters and
				persistent buffers.
		"""
		exclude_set = exclude_set or []
		own_state = self.state_dict()
		own_state_new = OrderedDict()

		if isinstance(exclude_set, str):
			exclude_set = [exclude_set]
		if isinstance(exclude_set, list):
			exclude_set_list = exclude_set
			exclude_set = lambda name: self.in_collection(name, exclude_set_list) or \
								self.remove_collection_name(name) in exclude_set_list
		if isinstance(whitelist, str):
			whitelist = [whitelist]
		if isinstance(whitelist, list):
			whitelist_list = whitelist
			whitelist = lambda name: self.in_collection(name, whitelist_list) or \
								self.remove_collection_name(name) in whitelist_list

		for name, param in own_state.items():
			if (exclude_set and exclude_set(name)) or (whitelist and not whitelist(name)):
				logging.warning('discard loading %s', name)
			else:
				own_state_new[self.remove_collection_name(name)] = param
		own_state = own_state_new

		state_dict_new = OrderedDict()
		for name, param in state_dict.items():
			state_dict_new[self.remove_collection_name(name)] = param
		state_dict = state_dict_new

		for name, param in state_dict.items():
			if name not in own_state:
				logging.warning('unexpected key %s in state_dict', name)
				continue
			if isinstance(param, Parameter):
				# backwards compatibility for serialized parameters
				param = param.data
			try:
				own_state[name].copy_(param)
			except:
				print('While copying the parameter named %s, whose dimensions in the model are'\
					  ' %s and whose dimensions in the checkpoint are %s, ...' %\
						  (name, own_state[name].size(), param.size()))
				raise

		missing = set(own_state.keys()) - set(state_dict.keys())
		if missing:
			logging.warning('missing keys in state_dict: %s', missing)
