import numpy as np
import logging
import time
import os
from itertools import chain

import torch
from torch import nn

from utils import Storage, BaseModel, SummaryHelper, storage_to_list, CheckpointManager, RAdam

from .model import Network

class LanguageModel(BaseModel):
	def __init__(self, param):
		args = param.args
		net = Network(param)
		self.optimizer = RAdam(net.get_parameters_by_name("", silent=True), lr=args.lr, weight_decay=1e-3)
		optimizerList = {"optimizer": self.optimizer}
		checkpoint_manager = CheckpointManager(args.name, args.model_dir, \
						args.checkpoint_steps, args.checkpoint_max_to_keep, "min")

		super().__init__(param, net, optimizerList, checkpoint_manager)

		self.create_summary()

	def create_summary(self):
		args = self.param.args
		self.summaryHelper = SummaryHelper("%s/%s_%s" % \
				(args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), \
				args)
		self.trainSummary = self.summaryHelper.addGroup(\
			scalar=["loss", "perplexity"],\
			prefix="train")

		scalarlist = ["loss", "loss0_ori", "loss0_adv", "loss1_ori", "loss1_adv", "loss0", "loss1",
			"perplexity", "perplexity0_ori", "perplexity0_adv", "perplexity1_ori", "perplexity1_adv", "perplexity0", "perplexity1"]
		tensorlist = []
		textlist = []
		emblist = []
		for i in self.args.show_sample:
			textlist.append("show_str%d" % i)
		self.devSummary = self.summaryHelper.addGroup(\
			scalar=scalarlist,\
			tensor=tensorlist,\
			text=textlist,\
			embedding=emblist,\
			prefix="dev")
		self.testSummary = self.summaryHelper.addGroup(\
			scalar=scalarlist,\
			tensor=tensorlist,\
			text=textlist,\
			embedding=emblist,\
			prefix="test")

	def _preprocess_batch(self, data):
		incoming = Storage()
		incoming.data = data = Storage(data)

		data.sent = torch.tensor(data.sent, dtype=torch.long, device=self.args.cuda)
		data.sent_length = np.array(data.sent_length)

		return incoming

	def get_next_batch(self, dm, key, restart=True):
		data = dm.get_next_batch(key)
		if data is None:
			if restart:
				dm.restart(key)
				return self.get_next_batch(dm, key, False)
			else:
				return None
		return self._preprocess_batch(data)

	def train(self, batch_num):
		args = self.param.args
		dm = self.param.volatile.dm
		datakey = 'train'
		dm.restart('train', args.batch_size)

		for i in range(batch_num):
			self.now_batch += 1

			self.zero_grad()
			incoming = self.get_next_batch(dm, datakey)
			incoming.args = Storage()
			self.net.forward(incoming)
			incoming.result.loss.backward()
			nn.utils.clip_grad_norm_(self.net.parameters(), args.grad_clip)
			self.optimizer.step()

			# incoming.result.lr = self.optimizer.param_groups[0]['lr']
			self.trainSummary(self.now_batch, storage_to_list(incoming.result))
			logging.info("batch %d : loss=%f", self.now_batch, incoming.result.loss)

	def predict_str(self, sent_str):
		incoming = Storage()
		incoming.data = data = Storage()
		data.batch_size = len(sent_str)
		data.sent_str = sent_str
		dm = self.param.volatile.dm
		# data.domain = torch.tensor(np.array([0 for _ in range(data.batch_size)]), dtype=torch.long, device=self.args.cuda)

		self.net.eval()
		self.net.forward(incoming)

		return incoming.result.losses.detach().numpy(), incoming.result.token_nums.detach().numpy()

	def evaluate(self, key):
		args = self.param.args
		dm = self.param.volatile.dm

		dm.restart(key, args.batch_size, shuffle=False)

		losses = []
		token_nums = []
		adv = []
		domain = []

		while True:
			incoming = self.get_next_batch(dm, key, restart=False)
			if incoming is None:
				break
			incoming.args = Storage()

			with torch.no_grad():
				self.net.forward(incoming)

				losses.extend(incoming.result.losses.detach().cpu().numpy())
				token_nums.extend(incoming.result.token_nums.detach().cpu().numpy())
				adv.extend(incoming.data.adv)
				domain.extend(incoming.data.domain)

		detail_arr = Storage()

		losses = np.array(losses)
		token_nums = np.array(token_nums)
		adv = np.array(adv)
		domain = np.array(domain)

		def calc(adv_mask, domain_mask, adv_name, domain_name):
			loss = np.sum(losses * adv_mask * domain_mask) / np.sum(token_nums * adv_mask * domain_mask)
			detail_arr[f"loss{domain_name}{adv_name}"] = loss
			detail_arr[f"perplexity{domain_name}{adv_name}"] = np.exp(detail_arr[f"loss{domain_name}{adv_name}"])
			return loss, np.exp(loss)

		calc(adv == 0, domain == 0, "_ori", "0")
		calc(adv == 0, domain == 1, "_ori", "1")
		calc(adv == 1, domain == 0, "_adv", "0")
		calc(adv == 1, domain == 1, "_adv", "1")
		calc(adv <= 1, domain == 0, "", "0")
		calc(adv <= 1, domain == 1, "", "1")
		calc(adv <= 1, domain <= 1, "", "")

		return detail_arr

	def train_process(self):
		args = self.param.args
		dm = self.param.volatile.dm

		while self.now_epoch < args.epochs:
			self.now_epoch += 1
			self.updateOtherWeights()

			self.net.train()
			self.train(args.batch_per_epoch)

			self.net.eval()
			# devloss_detail = self.evaluate("dev")
			# self.devSummary(self.now_batch, devloss_detail)
			# logging.info("epoch %d, evaluate dev", self.now_epoch)

			testloss_detail = self.evaluate("test")
			self.testSummary(self.now_batch, testloss_detail)
			logging.info("epoch %d, evaluate test", self.now_epoch)

			#self.lr_scheduler.step(devloss_detail.geo)
			self.save_checkpoint(value=testloss_detail.loss)

	def test(self, key):
		args = self.param.args
		dm = self.param.volatile.dm
		raise NotImplementedError("WIP")

	def test_process(self):
		logging.info("Test Start.")
		self.net.eval()
		#self.test("dev")
		test_res = self.test("test")
		logging.info("Test Finish.")
		return test_res
