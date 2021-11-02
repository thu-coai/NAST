# coding:utf-8
# modified from https://github.com/delldu/TextCNN

import logging

import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer

from utils import BaseModule, BaseNetwork, Storage

# pylint: disable=W0221
class Network(BaseNetwork):
	def __init__(self, param):
		super().__init__(param)

		self.featureLayer_train = FeatureExtractor(param)
		self.clsLayer_train = ClassifierLayer(param)

	def forward(self, incoming):
		incoming.result = Storage()

		self.featureLayer.forward(incoming)
		self.clsLayer.forward(incoming)

		if torch.isnan(incoming.result.loss).detach().cpu().numpy() > 0:
			logging.info("Nan detected")
			logging.info(incoming.result)
			raise FloatingPointError("Nan detected")

class FeatureExtractor(BaseModule):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param

		self.tokenizer = RobertaTokenizer.from_pretrained(args.roberta)
		self.roberta = RobertaModel.from_pretrained(args.roberta)

	def forward(self, incoming):
		incoming.hidden = hidden = Storage()

		tokenized_data = self.tokenizer(incoming.data.sent_str, return_tensors="pt", padding=True)
		tokenized_data["input_ids"] = tokenized_data["input_ids"].to(self.args.cuda)
		tokenized_data["attention_mask"] = tokenized_data["attention_mask"].to(self.args.cuda)
		hidden.hidden = self.roberta(**tokenized_data)[0]

class ClassifierLayer(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param

		self.fc1 = nn.Linear(768, 256)
		self.fc2 = nn.Linear(256, 1)
		self.drop = nn.Dropout(args.droprate)
		self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1]) * args.pos_weight)

	def forward(self, incoming):
		tmp = F.leaky_relu(self.fc1(incoming.hidden.hidden[:, 0]))
		logits = self.fc2(self.drop(tmp))[:, 0]

		incoming.result.loss = self.loss(logits, incoming.data.domain.float())
		incoming.result.predict = logits.sigmoid() > self.args.threshold
		incoming.result.acc = (incoming.result.predict.long() == incoming.data.domain).float().mean()
