# coding:utf-8
import logging
import time
import os
import shutil

import torch
from torch import nn, optim
import numpy as np
from torch.nn.utils import clip_grad_norm_

from cotk.metric import BleuCorpusMetric, MetricChain, LanguageGenerationRecorder
from utils.cotk_private.metric.name_changer import NameChanger
from utils import Storage, cuda, BaseModel, SummaryHelper, CheckpointManager, LongTensor, RAdam, gumbel_max, generateMask

from transformer import Network

def item(x, default_val=0):
	if x is not None:
		return x.item()
	else:
		return default_val

class Seq2seq(BaseModel):
	def __init__(self, param):
		args = param.args

		net = Network(param)

		if args.radam:
			adam = RAdam
		else:
			adam = optim.Adam

		self.optimizer_F = adam(net.get_parameters_by_name("F"), lr=args.lr_F, weight_decay=args.L2)
		self.optimizer_D = adam(net.get_parameters_by_name("D"), lr=args.lr_D, weight_decay=args.L2)

		optimizerList = {"optimizer_F": self.optimizer_F, "optimizer_D": self.optimizer_D}
		checkpoint_manager = CheckpointManager(args.name, args.model_dir, \
						args.checkpoint_steps, args.checkpoint_max_to_keep, ["max", "max"])

		super().__init__(param, net, optimizerList, checkpoint_manager)

		self.create_summary()

	def create_summary(self):
		args = self.param.args
		self.summaryHelper = SummaryHelper("%s/%s_%s" % \
				(args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), \
				args)
		self.trainSummary = self.summaryHelper.addGroup(\
			scalar=["slf_loss", "cyc_loss", "adv_loss", "d_adv_loss", "slf_length_loss", "cyc_length_loss", "slf_gen_error", "cyc_gen_error"], \
			prefix="train")

		scalarlist = ["fwbleu", "bwbleu", "fwsbleu", "bwsbleu", "fwgeo", "bwgeo", "fwacc", "bwacc", "fwhar", "bwhar",
				"fwlmppl", "bwlmppl", "fwacc", "bwacc", "fwsgeo", "bwsgeo", "fwshar", "bwshar", "fwoverall", "bwoverall",
				"fwldelta", "bwldelta"]
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

	def train_process(self):
		args = self.param.args

		if self.now_batch < args.F_pretrain_iter:
			print('Model F pretraining......')
			self.pretrain(args.F_pretrain_iter - self.now_batch)

		print('Training start......')
		while self.now_epoch < args.epochs:
			self.now_epoch += 1
			self.updateOtherWeights()

			self.train(args.eval_steps)

			devloss_detail = self.evaluate("dev", hasref=False, write_output=False)
			self.devSummary(self.now_batch, devloss_detail)
			logging.info("epoch %d, evaluate dev", self.now_epoch)

			testloss_detail = self.evaluate("test", hasref=True, write_output=True)
			self.testSummary(self.now_batch, testloss_detail)
			logging.info("epoch %d, evaluate test", self.now_epoch)

			flags = self.save_checkpoint(value=[devloss_detail.fwoverall, devloss_detail.bwoverall])

			output_dir = f"{self.args.out_dir}/{self.args.name}"
			if flags[0]:
				shutil.copyfile(f"{output_dir}/{self.now_batch}.neg2pos.txt",
		 			f'{output_dir}/best.neg2pos.txt')
			if flags[1]:
				shutil.copyfile(f"{output_dir}/{self.now_batch}.pos2neg.txt",
		 			f'{output_dir}/best.pos2neg.txt')

	def _preprocess_batch(self, data0, data1):
		incoming = Storage()
		incoming.data = data = Storage()

		if data0 is None:
			data0 = Storage({"sent": LongTensor(np.zeros((0, 0))), "ref_allvocabs": LongTensor(np.zeros((0, 0, 0))), "sent_length":[]})
		else:
			data0 = Storage(data0)
		if data1 is None:
			data1 = Storage({"sent": LongTensor(np.zeros((0, 0))), "ref_allvocabs": LongTensor(np.zeros((0, 0, 0))), "sent_length":[]})
		else:
			data1 = Storage(data1)

		data.batch_size = data0.sent.shape[0] + data1.sent.shape[0]
		data.seqlen = max(data0.sent.shape[1], data1.sent.shape[1]) - 1
		data.sent_length = np.concatenate([data0.sent_length, data1.sent_length], axis=0).astype(int)
		data.sent_length = LongTensor(data.sent_length) - 1
		data.sent = LongTensor(np.zeros((data.batch_size, data.seqlen), dtype=int))
		if data0.sent.shape[0]:
			data.sent[:data0.sent.shape[0], :data0.sent.shape[1] - 1] = cuda(torch.LongTensor(data0.sent[:, 1:])) # remove <go>
		if data1.sent.shape[0]:
			data.sent[data0.sent.shape[0]:, :data1.sent.shape[1] - 1] = cuda(torch.LongTensor(data1.sent[:, 1:])) # remove <go>
		data.domain = LongTensor([0 for _ in range(data0.sent.shape[0])] + [1 for _ in range(data1.sent.shape[0])])

		if "ref_allvocabs" in data0 and "ref_allvocabs" in data1:
			data.ref_allvocabs = LongTensor(np.zeros((data.batch_size,\
				max(data0.ref_allvocabs.shape[1], data1.ref_allvocabs.shape[1]), \
					max(data0.ref_allvocabs.shape[2], data1.ref_allvocabs.shape[2]) - 1), dtype=int))
			if data0.ref_allvocabs.shape[0]:
				data.ref_allvocabs[:data0.ref_allvocabs.shape[0], :data0.ref_allvocabs.shape[1], :data0.ref_allvocabs.shape[2] - 1] = cuda(torch.LongTensor(data0.ref_allvocabs)[:, :, 1:])
			if data1.ref_allvocabs.shape[0]:
				data.ref_allvocabs[data0.ref_allvocabs.shape[0]:, :data1.ref_allvocabs.shape[1], :data1.ref_allvocabs.shape[2] - 1] = cuda(torch.LongTensor(data1.ref_allvocabs)[:, :, 1:])

		return incoming

	def get_next_batch(self, dm, key, domain, restart=True, raw=False):
		old_key = key
		key = key + "_" + str(domain)

		data = dm.get_next_batch(key)
		if data is None:
			if restart:
				dm.restart(key)
				return self.get_next_batch(dm, old_key, domain, False, raw)
			else:
				return None
		if raw:
			return data
		elif domain == 0:
			return self._preprocess_batch(data, None)
		elif domain == 1:
			return self._preprocess_batch(None, data)

	def get_next_mix_batch(self, dm, key):
		data0 = self.get_next_batch(dm, key, 0, raw=True)
		data1 = self.get_next_batch(dm, key, 1, raw=True)
		return self._preprocess_batch(data0, data1)

	def pretrain(self, batch_num):
		# pretrain F model using slf_loss

		self.net.model_F.train()
		self.net.model_D.train()

		args = self.param.args
		dm = self.param.volatile.dm
		datakey = 'train'
		dm.restart('train_0', args.batch_size)
		dm.restart('train_1', args.batch_size)

		his_f_slf_loss = []
		his_f_cyc_loss = []
		his_f_slf_length_loss = []

		for i in range(batch_num):
			self.now_batch += 1

			incoming = self.get_next_mix_batch(dm, datakey)

			slf_loss, cyc_loss, _, slf_length_loss, _, slf_gen_error, _ = self.f_step(incoming, temperature=1.0, cyc_rec_enable=False, use_inference_z=True)
			his_f_slf_loss.append(slf_loss)
			his_f_cyc_loss.append(cyc_loss)
			his_f_slf_length_loss.append(slf_length_loss)

			if (i + 1) % 10 == 0:
				avrg_f_slf_loss = np.mean(his_f_slf_loss)
				avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
				avrg_f_slf_length = np.mean(his_f_slf_length_loss)
				his_f_slf_loss = []
				his_f_cyc_loss = []
				his_f_slf_length_loss = []
				print('[iter: {}] slf_loss:{:.4f}, rec_loss:{:.4f}, slf_length_loss:{:.4f}'.format(i + 1, avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_slf_length))

				self.trainSummary(self.now_batch, {"slf_loss": avrg_f_slf_loss, "cyc_loss": avrg_f_cyc_loss, "slf_length_loss": avrg_f_slf_length})

	def calc_temperature(self, temperature_config, step):
		num = len(temperature_config)
		for i in range(num):
			t_a, s_a = temperature_config[i]
			if i == num - 1:
				return t_a
			t_b, s_b = temperature_config[i + 1]
			if s_a <= step < s_b:
				k = (step - s_a) / (s_b - s_a)
				temperature = (1 - k) * t_a + k * t_b
				return temperature

	def train(self, batch_num):
		self.net.model_F.train()
		self.net.model_D.train()

		args = self.param.args
		dm = self.param.volatile.dm
		datakey = 'train'
		dm.restart('train_0', args.batch_size)
		dm.restart('train_1', args.batch_size)

		his_d_adv_loss = []
		his_f_slf_loss = []
		his_f_cyc_loss = []
		his_f_adv_loss = []
		his_f_slf_length_loss = []
		his_f_cyc_length_loss = []
		his_f_slf_gen_error = []
		his_f_cyc_gen_error = []

		for i in range(batch_num):
			self.now_batch += 1

			temperature = self.calc_temperature(args.temperature_config, self.now_batch)

			for _ in range(args.iter_D):
				incoming = self.get_next_mix_batch(dm, datakey)
				d_adv_loss = self.d_step(incoming, temperature=temperature, use_inference_z=(self.now_batch-args.F_pretrain_iter) > args.pre_z_batch)
				his_d_adv_loss.append(d_adv_loss)

			for _ in range(args.iter_F):
				incoming = self.get_next_mix_batch(dm, datakey)
				f_slf_loss, f_cyc_loss, f_adv_loss, f_slf_length_loss, f_cyc_length_loss, f_slf_gen_error, f_cyc_gen_error = self.f_step(incoming,
					temperature=temperature, use_inference_z=(self.now_batch-args.F_pretrain_iter) > args.pre_z_batch
				)
				his_f_slf_loss.append(f_slf_loss)
				his_f_cyc_loss.append(f_cyc_loss)
				his_f_adv_loss.append(f_adv_loss)
				his_f_slf_length_loss.append(f_slf_length_loss)
				his_f_cyc_length_loss.append(f_cyc_length_loss)
				his_f_slf_gen_error.append(f_slf_gen_error)
				his_f_cyc_gen_error.append(f_cyc_gen_error)

			if self.now_batch % args.log_steps == 0:
				avrg_d_adv_loss = np.mean(his_d_adv_loss)
				avrg_f_slf_loss = np.mean(his_f_slf_loss)
				avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
				avrg_f_adv_loss = np.mean(his_f_adv_loss)
				avrg_f_slf_length_loss = np.mean(his_f_slf_length_loss)
				avrg_f_cyc_length_loss = np.mean(his_f_cyc_length_loss)
				avrg_f_slf_gen_error = np.mean(his_f_slf_gen_error)
				avrg_f_cyc_gen_error = np.mean(his_f_cyc_gen_error)
				log_str = '[iter {}] d_adv_loss: {:.4f}  ' + \
						'f_slf_loss: {:.4f}  f_cyc_loss: {:.4f}  ' + \
						'f_adv_loss: {:.4f}  f_slf_length_loss: {:.4f}  ' + \
						'f_cyc_length_loss: {:.4f}  temp: {:.4f}  ' + \
						'f_slf_gen_error: {:.4f} f_cyc_gen_error: {:.4f}'
				print(log_str.format(
					self.now_batch, avrg_d_adv_loss,
					avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_adv_loss,
					avrg_f_slf_length_loss, avrg_f_cyc_length_loss,
					temperature, avrg_f_slf_gen_error, avrg_f_cyc_gen_error
				))

				self.trainSummary(self.now_batch, {"d_adv_loss": avrg_d_adv_loss, "cyc_loss": avrg_f_cyc_loss, \
					"adv_loss": avrg_f_adv_loss, "slf_loss": avrg_f_slf_loss, "slf_length_loss": avrg_f_slf_length_loss, \
					'cyc_length_loss': avrg_f_cyc_length_loss, "slf_gen_error": avrg_f_slf_gen_error, "cyc_gen_error": avrg_f_cyc_gen_error})

	def evaluate(self, datakey, hasref, write_output):
		dm = self.param.volatile.dm
		self.net.model_F.eval()
		vocab_size = dm.frequent_vocab_size
		eos_idx = dm.eos_id

		temperature = self.calc_temperature(self.args.temperature_config, self.now_batch)

		def inference(datakey, domain):
			dm.restart(f"{datakey}_{domain}", self.args.batch_size, shuffle=False)

			metric = MetricChain()
			if hasref:
				metric.add_metric(BleuCorpusMetric(dm, 4, reference_num=4))
			metric.add_metric(NameChanger(LanguageGenerationRecorder(dm, "sref_allvocabs"), "s"))
			metric.add_metric(LanguageGenerationRecorder(dm))
			metric.add_metric(NameChanger(BleuCorpusMetric(dm, 4, reference_allvocabs_key="sref_allvocabs"), "s"))
			predict_res = []
			length_res = []
			allpos_res = []

			while True:
				incoming = self.get_next_batch(dm, datakey, domain, restart=False)
				if incoming is None:
					break

				inp_tokens = incoming.data.sent
				inp_lengths = incoming.data.sent_length
				raw_styles = torch.full_like(inp_tokens[:, 0], domain)
				rev_styles = 1 - raw_styles

				with torch.no_grad():
					rev_log_probs, gen_lengths, _, _, allpos = self.net.model_F(
						inp_tokens,
						inp_lengths,
						None,
						None,
						rev_styles,
						generate=True,
						differentiable_decode=False,
						temperature=temperature,
						use_inference_z = (self.now_batch-self.args.F_pretrain_iter) > self.args.pre_z_batch
					)

				data = Storage()
				data.gen = rev_log_probs.argmax(-1).detach().cpu().numpy()
				if hasref:
					data.ref_allvocabs = incoming.data.ref_allvocabs.detach().cpu().numpy()
				data.sref_allvocabs = incoming.data.sent.detach().cpu().numpy()
				metric.forward(data)

				sents = [dm.convert_ids_to_sentence(data.gen[i]) for i in range(incoming.data.batch_size)]
				predict_res.append(self.param.volatile.cls.predict_str(sents))

				if self.args.use_learnable:
					length_res.append((gen_lengths - inp_lengths).detach().cpu().numpy())
					for glen, pos in zip(gen_lengths.detach().cpu().numpy(), allpos.detach().cpu().numpy()):
						allpos_res.append(pos[1:glen + 1])

			result = metric.close()
			result["acc"] = np.mean(np.concatenate(predict_res) == 1 - domain)
			if self.args.use_learnable:
				result["length_delta"] = np.mean(np.concatenate(length_res))
			else:
				result["length_delta"] = 0

			show_str = []
			for i in range(self.args.batch_size):
				show_str.append(f"sent{domain}:" + result["sgen"][i])
				show_str.append(f"gen{domain}:" + result["gen"][i])
			result["show_str"] = show_str
			result["allpos"] = allpos_res
			return result

		result0 = inference(datakey, 0)
		result1 = inference(datakey, 1)


		detail_arr = Storage()
		detail_arr["fwacc"] = result0["acc"]
		detail_arr["bwacc"] = result1["acc"]
		detail_arr["fwsbleu"] = result0["sbleu"]
		detail_arr["bwsbleu"] = result1["sbleu"]
		detail_arr["fwsgeo"] = np.sqrt(result0["sbleu"] * result0["acc"])
		detail_arr["bwsgeo"] = np.sqrt(result1["sbleu"] * result1["acc"])
		detail_arr["fwshar"] = 2 * result0["sbleu"] * result0["acc"] / (result0["sbleu"] + result0["acc"] + 1e-6)
		detail_arr["bwshar"] = 2 * result1["sbleu"] * result1["acc"] / (result1["sbleu"] + result1["acc"] + 1e-6)
		detail_arr["fwldelta"] = result0["length_delta"]
		detail_arr["bwldelta"] = result1["length_delta"]

		def clipoverall(acc, bleu, acc_limit=0.8, bleu_limit=0.4):
			# clip the model selection criteria to avoid too low accuracy or bleu
			acc_clip = max(acc - acc_limit, 0)
			bleu_clip = max(bleu - bleu_limit, 0)
			return np.sqrt(acc_clip * bleu_clip)

		if hasref:
			detail_arr["fwbleu"] = result0["bleu"]
			detail_arr["bwbleu"] = result1["bleu"]
			detail_arr["fwgeo"] = np.sqrt(result0["bleu"] * result0["acc"])
			detail_arr["bwgeo"] = np.sqrt(result1["bleu"] * result1["acc"])
			detail_arr["fwhar"] = 2 * result0["bleu"] * result0["acc"] / (result0["bleu"] + result0["acc"] + 1e-6)
			detail_arr["bwhar"] = 2 * result1["bleu"] * result1["acc"] / (result1["bleu"] + result1["acc"] + 1e-6)

			detail_arr["fwoverall"] = clipoverall(result0["acc"], result0["bleu"])
			detail_arr["bwoverall"] = clipoverall(result1["acc"], result1["bleu"])
		else:
			detail_arr["fwoverall"] = clipoverall(result0["acc"], result0["sbleu"])
			detail_arr["bwoverall"] = clipoverall(result1["acc"], result1["sbleu"])

		detail_arr["show_str0"] = "\n".join(result0["show_str"] + result1["show_str"])

		if write_output:
			output_dir = f"{self.args.out_dir}/{self.args.name}"
			os.makedirs(output_dir, exist_ok=True)
			with open(f"{output_dir}/{self.now_batch}.neg2pos.txt", 'w') as f:
				for sent in result0["gen"]:
					f.write(sent + "\n")

			with open(f"{output_dir}/{self.now_batch}.pos2neg.txt", 'w') as f:
				for sent in result1["gen"]:
					f.write(sent + "\n")

		return detail_arr

	def approx(self, logits, mode):
		if mode == "soft":
			soft_tokens = logits.exp()
		elif mode == "gumbel":
			soft_tokens = gumbel_max(logits)[1]
		else:
			raise NotImplementedError()
		return soft_tokens

	def f_step(self, incoming, temperature, cyc_rec_enable=True, *, use_inference_z):

		self.net.model_D.eval()

		dm = self.param.volatile.dm

		pad_idx = dm.pad_id
		eos_idx = dm.eos_id

		vocab_size = dm.frequent_vocab_size
		loss_fn = nn.NLLLoss(reduction='none')

		inp_tokens = incoming.data.sent
		inp_lengths = incoming.data.sent_length
		raw_styles = incoming.data.domain
		rev_styles = 1 - raw_styles
		batch_size = inp_tokens.size(0)
		token_mask = (inp_tokens != pad_idx).float()

		self.optimizer_F.zero_grad()

		args = self.args
		noise_inp_tokens, noise_inp_lengths = self.net.noiseLayer(inp_tokens.transpose(0, 1), inp_lengths, rev_domain=rev_styles)
		noise_inp_tokens = noise_inp_tokens.transpose(0, 1)

		slf_log_probs, _, slf_length_losses, slf_gen_error, _ = self.net.model_F(
			noise_inp_tokens,
			noise_inp_lengths,
			inp_tokens,
			inp_lengths,
			raw_styles,
			generate=False,
			differentiable_decode=False,
			temperature=temperature,
			use_inference_z=use_inference_z
		)

		slf_rec_loss = loss_fn(slf_log_probs.transpose(1, 2), inp_tokens) * token_mask
		slf_rec_loss = slf_rec_loss.sum() / batch_size
		slf_rec_loss *= self.args.slf_factor

		if args.use_learnable:
			if len(slf_length_losses.shape) <= 1:
				slf_rec_length_loss = slf_length_losses
			else:
				length_mask = generateMask(slf_length_losses.shape[1], inp_lengths.detach().cpu().numpy() + 1, type=float, device=slf_length_losses.device).transpose(0, 1)
				slf_rec_length_loss = (slf_length_losses * length_mask).sum() / batch_size * self.args.length_factor

			(slf_rec_loss + slf_rec_length_loss).backward()
		else:
			slf_rec_length_loss = None
			slf_rec_loss.backward()

		# cycle consistency loss

		if not cyc_rec_enable:
			self.optimizer_F.step()
			self.net.model_D.train()
			return slf_rec_loss.item(), 0, 0, item(slf_rec_length_loss), 0, item(slf_gen_error), 0

		gen_log_probs, gen_lengths, _, _, _ = self.net.model_F(
			inp_tokens,
			inp_lengths,
			None,
			None,
			rev_styles,
			generate=True,
			differentiable_decode=True,
			temperature=temperature,
			use_inference_z=use_inference_z
		)

		gen_soft_tokens = self.approx(gen_log_probs, self.args.gen_approx_mode)
		if not args.use_learnable:
			gen_lengths = self.get_lengths(gen_soft_tokens.argmax(-1), eos_idx)

		cyc_log_probs, _, cyc_length_losses, cyc_gen_error, _ = self.net.model_F(
			gen_soft_tokens,
			gen_lengths,
			inp_tokens,
			inp_lengths,
			raw_styles,
			generate=False,
			differentiable_decode=False,
			temperature=temperature,
			use_inference_z=use_inference_z
		)

		cyc_rec_loss = loss_fn(cyc_log_probs.transpose(1, 2), inp_tokens) * token_mask
		cyc_rec_loss = cyc_rec_loss.sum() / batch_size
		cyc_rec_loss *= self.args.cyc_factor

		if args.use_learnable:
			if len(cyc_length_losses.shape) <= 1:
				cyc_rec_length_loss = cyc_length_losses
			else:
				length_mask = generateMask(cyc_length_losses.shape[1], inp_lengths.detach().cpu().numpy() + 1, type=float, device=cyc_length_losses.device).transpose(0, 1)
				cyc_rec_length_loss = (cyc_length_losses * length_mask).sum() / batch_size * self.args.length_factor
		else:
			cyc_rec_length_loss = None

		# style consistency loss

		adv_log_porbs = self.net.model_D(gen_soft_tokens, gen_lengths, rev_styles)
		if self.args.discriminator_method == 'Multi':
			adv_labels = rev_styles + 1
		else:
			adv_labels = torch.ones_like(rev_styles)
		adv_loss = loss_fn(adv_log_porbs, adv_labels)
		adv_factor = (raw_styles == 0) * self.args.fw_adv_factor + (raw_styles == 1) * self.args.bw_adv_factor
		adv_loss = (adv_loss * adv_factor).sum() / batch_size
		# adv_loss *= self.args.adv_factor

		if args.use_learnable:
			(cyc_rec_loss + adv_loss + cyc_rec_length_loss).backward()
		else:
			(cyc_rec_loss + adv_loss).backward()

		# update parameters

		clip_grad_norm_(self.net.model_F.parameters(), 5)
		self.optimizer_F.step()

		self.net.model_D.train()

		return slf_rec_loss.item(), cyc_rec_loss.item(), adv_loss.item(), \
			item(slf_rec_length_loss), item(cyc_rec_length_loss), \
			item(slf_gen_error), item(cyc_gen_error)

	def get_lengths(self, tokens, eos_idx):
		lengths = torch.cumsum(tokens == eos_idx, 1)
		lengths = (lengths == 0).long().sum(-1)
		lengths = torch.min(lengths + 1, torch.ones_like(lengths) * tokens.shape[1]) # +1 for <eos> token
		return lengths

	def d_step(self, incoming, temperature, *, use_inference_z):
		dm = self.param.volatile.dm

		self.net.model_F.eval()
		pad_idx = dm.pad_id
		eos_idx = dm.eos_id
		vocab_size = dm.frequent_vocab_size
		loss_fn = nn.NLLLoss(reduction='none')

		inp_tokens = incoming.data.sent
		inp_lengths = incoming.data.sent_length
		raw_styles = incoming.data.domain
		# inp_tokens: batch_size * length (with posi and nega)
		# inp_length
		# raw_styles
		rev_styles = 1 - raw_styles
		batch_size = inp_tokens.size(0)

		with torch.no_grad():
			raw_gen_log_probs, raw_gen_lengths, _, _, _ = self.net.model_F(
				inp_tokens,
				inp_lengths,
				None,
				None,
				raw_styles,
				generate=True,
				differentiable_decode=True,
				temperature=temperature,
				use_inference_z=use_inference_z
			)
			rev_gen_log_probs, rev_gen_lengths, _, _, _ = self.net.model_F(
				inp_tokens,
				inp_lengths,
				None,
				None,
				rev_styles,
				generate=True,
				differentiable_decode=True,
				temperature=temperature,
				use_inference_z=use_inference_z
			)

		raw_gen_soft_tokens = self.approx(raw_gen_log_probs, self.args.dis_approx_mode)
		if not self.args.use_learnable:
			raw_gen_lengths = self.get_lengths(raw_gen_soft_tokens.argmax(-1), eos_idx)

		rev_gen_soft_tokens = self.approx(rev_gen_log_probs, self.args.dis_approx_mode)
		if not self.args.use_learnable:
			rev_gen_lengths = self.get_lengths(rev_gen_soft_tokens.argmax(-1), eos_idx)

		if self.args.discriminator_method == 'Multi':
			gold_log_probs = self.net.model_D(inp_tokens, inp_lengths)
			gold_labels = raw_styles + 1

			raw_gen_log_probs = self.net.model_D(raw_gen_soft_tokens, raw_gen_lengths)
			rev_gen_log_probs = self.net.model_D(rev_gen_soft_tokens, rev_gen_lengths)
			gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
			raw_gen_labels = raw_styles + 1
			rev_gen_labels = torch.zeros_like(rev_styles)
			gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)
		else:
			raw_gold_log_probs = self.net.model_D(inp_tokens, inp_lengths, raw_styles)
			rev_gold_log_probs = self.net.model_D(inp_tokens, inp_lengths, rev_styles)
			gold_log_probs = torch.cat((raw_gold_log_probs, rev_gold_log_probs), 0)
			raw_gold_labels = torch.ones_like(raw_styles)
			rev_gold_labels = torch.zeros_like(rev_styles)
			gold_labels = torch.cat((raw_gold_labels, rev_gold_labels), 0)


			raw_gen_log_probs = self.net.model_D(raw_gen_soft_tokens, raw_gen_lengths, raw_styles)
			rev_gen_log_probs = self.net.model_D(rev_gen_soft_tokens, rev_gen_lengths, rev_styles)
			gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
			raw_gen_labels = torch.ones_like(raw_styles)
			rev_gen_labels = torch.zeros_like(rev_styles)
			gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)


		adv_log_probs = torch.cat((gold_log_probs, gen_log_probs), 0)
		adv_labels = torch.cat((gold_labels, gen_labels), 0)
		adv_loss = loss_fn(adv_log_probs, adv_labels)
		assert len(adv_loss.size()) == 1
		adv_loss = adv_loss.sum() / batch_size
		loss = adv_loss

		self.optimizer_D.zero_grad()
		loss.backward()
		clip_grad_norm_(self.net.model_D.parameters(), 5)
		self.optimizer_D.step()

		self.net.model_F.train()

		return adv_loss.item()

	def test_process(self):
		self.evaluate("test", hasref=True)