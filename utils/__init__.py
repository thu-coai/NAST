# -*- coding: utf-8 -*-

from .anneal_helper import AnnealHelper, AnnealParameter
from .debug_helper import debug
from .cache_helper import try_cache
from .storage import Storage
from .summaryx_helper import SummaryHelper
from .cuda_helper import cuda, zeros, ones, Tensor, LongTensor
from .cuda_helper import init as cuda_init
from .model_helper import BaseModel, get_mean, storage_to_list
from .network_helper import BaseNetwork
from .module_helper import BaseModule
from .scheduler_helper import ReduceLROnLambda
from .checkpoint_helper import CheckpointManager, name_format
from .optimizer import AdamW, RAdam
from .operator_helper import reshape, cdist_nobatch, sequence_pooling, broadcast, unsqueeze, tensor, longtensor, onet, zerot, generateMask
from .gumbel import gumbel_max, gumbel_max_with_mask, gumbel_softmax, recenter_gradient, RebarGradient, straight_max, cdist, onehot

import numpy as np
def padding_id(data, pad_value=0, dtype=int): # seqlen * batch
	batch_size = len(data)
	length = [len(d) for d in data]
	seqlen = max(length)

	res = np.ones((seqlen, batch_size), dtype=dtype) * pad_value
	for i, d in enumerate(data):
		res[:len(d), i] = d
	return res, length