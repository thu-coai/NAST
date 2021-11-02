import numpy as np
import logging
import time
import os
from itertools import chain

import torch
from torch import nn

from utils import Storage, BaseModel, SummaryHelper, storage_to_list, CheckpointManager, RAdam

from .model import Network

class Classifier(BaseModel):
	def __init__(self, param):
		args = param.args
		net = Network(param)
		self.optimizer = RAdam(net.get_parameters_by_name("train", silent=True), lr=args.lr, weight_decay=1e-3)
		optimizerList = {"optimizer": self.optimizer}
		checkpoint_manager = CheckpointManager(args.name, args.model_dir, \
						args.checkpoint_steps, args.checkpoint_max_to_keep, "max")

		super().__init__(param, net, optimizerList, checkpoint_manager)

		self.create_summary()

	def create_summary(self):
		args = self.param.args
		self.summaryHelper = SummaryHelper("%s/%s_%s" % \
				(args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), \
				args)
		self.trainSummary = self.summaryHelper.addGroup(\
			scalar=["loss", "acc", "acc0", "acc1", "f1", "acc_ori", "acc0_ori", "acc1_ori", "f1_ori", "acc_ref", "acc0_ref", "acc1_ref", "f1_ref"],\
			prefix="train")

		scalarlist = ["loss", "acc", "acc0", "acc1", "f1", "acc_ori", "acc0_ori", "acc1_ori", "f1_ori", "acc_ref", "acc0_ref", "acc1_ref", "f1_ref"]
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
		data.domain = torch.tensor(data.domain, dtype=torch.long, device=self.args.cuda)
		data.adv = torch.tensor(data.adv, dtype=torch.long, device=self.args.cuda)
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
		data.domain = torch.tensor(np.array([0 for _ in range(data.batch_size)]), dtype=torch.long, device=self.args.cuda)

		self.net.eval()
		self.net.forward(incoming)

		return incoming.result.predict.detach().cpu().numpy()

	def evaluate(self, key):
		args = self.param.args
		dm = self.param.volatile.dm

		dm.restart(key, args.batch_size, shuffle=False)

		answer = []
		predict = []
		adv = []

		while True:
			incoming = self.get_next_batch(dm, key, restart=False)
			if incoming is None:
				break
			incoming.args = Storage()

			with torch.no_grad():
				self.net.forward(incoming)

				now_answer = incoming.data.domain.detach().cpu().numpy()
				now_predict = incoming.result.predict.detach().cpu().numpy()
				now_adv = incoming.data.adv.detach().cpu().numpy()

				answer.append(now_answer)
				predict.append(now_predict)
				adv.append(now_adv)

		def calcacc(answer, predict):
			acc = np.mean(answer == predict)
			acc0 = np.sum((answer == 0) * (predict == 0)) / np.sum(answer == 0)
			acc1 = np.sum((answer == 1) * (predict == 1)) / np.sum(answer == 1)
			f1 = acc0 * acc1 * 2 / (acc0 + acc1 + 1e-10)
			return acc, acc0, acc1, f1

		answer = np.concatenate(answer, axis=0)
		predict = np.concatenate(predict, axis=0)
		adv = np.concatenate(adv, axis=0)

		detail_arr = Storage()
		detail_arr["acc"], detail_arr["acc0"], detail_arr["acc1"], detail_arr["f1"] = calcacc(answer, predict)
		answer_ori = answer[adv == 0]
		predict_ori = predict[adv == 0]
		detail_arr["acc_ori"], detail_arr["acc0_ori"], detail_arr["acc1_ori"], detail_arr["f1_ori"] = calcacc(answer_ori, predict_ori)
		answer_ref = answer[adv == 1]
		predict_ref = predict[adv == 1]
		detail_arr["acc_ref"], detail_arr["acc0_ref"], detail_arr["acc1_ref"], detail_arr["f1_ref"] = calcacc(answer_ref, predict_ref)

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
			self.save_checkpoint(value=testloss_detail.f1)

	def test(self, key):
		args = self.param.args
		dm = self.param.volatile.dm

		testloss_detail = self.evaluate("test")
		self.testSummary(self.now_batch, testloss_detail)
		logging.info("epoch %d, evaluate test", self.now_epoch)

	def test_process(self):
		logging.info("Test Start.")
		self.net.eval()
		#self.test("dev")
		self.test("test")
		logging.info("Test Finish.")
