# coding:utf-8
# modified from https://github.com/delldu/TextCNN

import logging

import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from utils import BaseNetwork, Storage

# pylint: disable=W0221
class Network(BaseNetwork):
	def __init__(self, param):
		super().__init__(param)

		args = param.args

		self.tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
		self.tokenizer.pad_token = self.tokenizer.eos_token
		self.tokenizer.cls_token = self.tokenizer.eos_token
		self.tokenizer.mask_token = self.tokenizer.eos_token
		self.tokenizer.sep_token = self.tokenizer.eos_token
		#self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
		self.gpt2 = GPT2LMHeadModel.from_pretrained(args.gpt2)
		self.lossCE = torch.nn.CrossEntropyLoss(reduction="none")

	def forward(self, incoming):
		incoming.result = result = Storage()

		tokenized_data = self.tokenizer(incoming.data.sent_str, return_tensors="pt", padding=True)
		input_ids = tokenized_data["input_ids"].to(self.args.cuda)
		if input_ids.shape[1] == 0:
			batch_size = len(incoming.data.sent_str)
			result.loss = torch.zeros(1, device=self.args.cuda)[0]
			result.losses = torch.zeros(batch_size, device=self.args.cuda)
			result.token_nums = torch.zeros(1, dtype=torch.long, device=self.args.cuda)[0]
			return

		input_ids = torch.cat([torch.ones_like(input_ids[:, :1]) * self.tokenizer.eos_token_id, input_ids], dim=1)
		attention_mask = tokenized_data["attention_mask"].to(self.args.cuda)
		attention_mask = torch.cat([torch.ones_like(attention_mask[:, :1]), attention_mask], dim=1)
		labels = input_ids.clone()
		labels.masked_fill_(~attention_mask.bool(), -100)
		loss, logits = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)[:2]

		result.loss = loss
		losses = self.lossCE(logits.transpose(1, 2), torch.cat([labels[:, 1:], torch.ones_like(labels[:, :1]) * -100], dim=-1))
		result.losses = losses.sum(dim=-1)
		result.token_nums = tokenized_data["attention_mask"].sum(dim=-1)
		# result.losses = outputs.
		# logits = outputs.logits

		if torch.isnan(incoming.result.loss).detach().cpu().numpy() > 0:
			logging.info("Nan detected")
			logging.info(incoming.result)
			raise FloatingPointError("Nan detected")
