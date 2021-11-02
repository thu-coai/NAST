# coding:utf-8
import os
import logging
import json
from collections import Counter, OrderedDict
from itertools import product
import copy
import numpy as np
import json

from cotk.metric import MetricChain, BleuCorpusMetric
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import debug, try_cache, cuda_init, Storage, padding_id
from utils.cotk_private.dataloader.predefined_language_generation import PredefinedStyleTransfer
from utils.cotk_private.metric.name_changer import NameChanger
from classifier import Classifier
from languagemodel import LanguageModel

from itertools import zip_longest
import torch

def zip_equal(*iterables):
	sentinel = object()
	for combo in zip_longest(*iterables, fillvalue=sentinel):
		if sentinel in combo:
			raise ValueError('Iterables have different lengths')
		yield combo

def run(*argv):
	import argparse
	import time

	from utils import Storage

	parser = argparse.ArgumentParser(description='evaluation code')
	args = Storage()

	parser.add_argument('--dataid', type=str, default='../data/yelp')
	parser.add_argument('--dev0', type=str, default=None)
	parser.add_argument('--dev1', type=str, default=None)
	parser.add_argument('--test0', type=str, default=None)
	parser.add_argument('--test1', type=str, default=None)
	parser.add_argument('--output', type=str, default="result.txt")
	parser.add_argument('--clsrestore', type=str, default="cls_yelp_best")
	parser.add_argument('--lmrestore', type=str, default="lm_yelp_best")
	parser.add_argument('--allow_unk', action="store_true")

	parser.add_argument('--cache', action='store_true',
		help='Cache the dataloader')
	parser.add_argument('--debug', action='store_true',
		help='Enter debug mode (using ptvsd).')
	parser.add_argument('--seed', type=int, default=0,
		help='Specify random seed. Default: 0')

	parser.add_argument('--name', type=str, default=None)

	cargs = parser.parse_args(argv)
		# Editing following arguments to bypass command line.

	cuda_init(0, True)

	args.dataid = cargs.dataid
	args.dev0 = cargs.dev0
	args.dev1 = cargs.dev1
	args.test0 = cargs.test0
	args.test1 = cargs.test1
	args.debug = cargs.debug
	args.cache = cargs.cache
	args.seed = cargs.seed
	args.clsrestore = cargs.clsrestore
	args.output = cargs.output
	args.allow_unk = cargs.allow_unk
	args.lmrestore = cargs.lmrestore

	if args.dev0 is None and args.dev1 is None and args.test0 is None and args.test1 is None:
		args.test0 = f"./output/{cargs.name}/best.neg2pos.txt"
		args.test1 = f"./output/{cargs.name}/best.pos2neg.txt"

	import random
	random.seed(cargs.seed)
	import torch
	torch.manual_seed(cargs.seed)
	import numpy as np
	np.random.seed(cargs.seed)

	eval_main(args)

def eval_main(args):
	logging.basicConfig(\
		filename=0,\
		level=logging.DEBUG,\
		format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',\
		datefmt='%H:%M:%S')

	if args.debug:
		debug()

	data_class = PredefinedStyleTransfer
	data_arg = Storage()
	data_arg.file_id = args.dataid
	data_arg.max_sent_length = None
	data_arg.fields = {
					"train_0": OrderedDict([("sent", "SentenceDefault")]),
					"train_1": OrderedDict([("sent", "SentenceDefault")]),
					"dev_0": OrderedDict([("sent", "SentenceDefault")]),
					"dev_1": OrderedDict([("sent", "SentenceDefault")]),
					"test_0": OrderedDict([("sent", "SentenceDefault"), ("ref", "SessionDefault")]),
					"test_1": OrderedDict([("sent", "SentenceDefault"), ("ref", "SessionDefault")]),
				}

	def load_dataset(data_arg):
		dm = data_class(**data_arg)
		return dm

	if args.cache:
		dm = try_cache(load_dataset, (data_arg,),
			"./cache", "eval" + data_class.__name__)
	else:
		dm = load_dataset(data_arg)

	import run_cls
	cls_param = Storage()
	cls_param.args = run_cls.run("--dryrun", "--restore", args.clsrestore, "--cuda", "--dataid", args.dataid)
	cls_param.volatile = Storage()
	cls_param.volatile.load_exclude_set = []
	cls_param.volatile.restoreCallback = None
	cls_param.volatile.dm = dm
	classifier = Classifier(cls_param)

	import run_lm
	lm_param = Storage()
	lm_param.args = run_lm.run("--dryrun", "--restore", args.lmrestore, "--dataid", args.dataid)
	lm_param.volatile = Storage()
	lm_param.volatile.load_exclude_set = []
	lm_param.volatile.restoreCallback = None
	lm_param.volatile.dm = dm
	lm = LanguageModel(lm_param)

	# read target file
	def read_target(filename):
		sent_str = []
		sent = []
		unk_num = 0
		with open(filename, "r") as f:
			for line in f:
				sstr = line.strip().lower()
				sent_str.append(sstr)
				sent_id = dm.convert_sentence_to_ids(sstr)
				if dm.unk_id in sent_id:
					unk_num += 1
					print("unk detected")
					print(sstr)
					print(dm.convert_ids_to_sentence(sent_id))
					if not args.allow_unk:
						raise RuntimeError("unk in generated file")
				sent.append(sent_id)
		print("unk percent:%.4f" % (unk_num / len(sent)))
		return sent_str, sent

	def calc(filename, field_key, domain, haveref=True):
		sent_str, sent = read_target(filename)

		# acc
		predict_class = []
		for chunk_str in [sent_str[i:i + 64] for i in range(0, len(sent_str), 64)]:
			predict_class.extend(classifier.predict_str(chunk_str))
		acc = (np.array(predict_class) == 1 - domain).astype(float).mean()

		# ppl
		losses_arr = []
		token_nums_arr = []
		for chunk_str in [sent_str[i:i + 64] for i in range(0, len(sent_str), 64)]:
			losses, target_nums = lm.predict_str(chunk_str)
			losses_arr.append(losses)
			token_nums_arr.append(target_nums)

		metric = MetricChain()
		# metric.add_metric(NgramPerplexityMetric(dm, dm.get_all_batch(f"train_{1-domain}")['sent_allvocabs'], 4, gen_key="sent"))
		if haveref:
			metric.add_metric(BleuCorpusMetric(dm, 4, reference_num=4))
		metric.add_metric(NameChanger(BleuCorpusMetric(dm, 4, reference_num=1, reference_allvocabs_key="sent_allvocabs"), "self"))

		for data, chunk_sent in zip_equal(dm.get_batches(f"{field_key}_{domain}", 64, shuffle=False),\
					[sent[i:i + 64] for i in range(0, len(sent), 64)]):
			data["gen"] = padding_id(chunk_sent)[0].transpose(1, 0)
			metric.forward(data)

		mres = Storage(metric.close())

		res = Storage()
		res.acc = acc
		res.self_bleu = mres.selfbleu
		res.ppl = np.exp(np.sum(np.concatenate(losses_arr)) / np.sum(np.concatenate(token_nums_arr)))

		def getmean(*param):
			summ = np.array(param) + 1e-10
			g2 = np.exp(np.mean(np.log(summ)))
			h2 = np.exp(np.sum(np.log(summ))) * len(summ) / np.sum(summ)
			return g2, h2

		res.self_g2, res.self_h2 = getmean(res.acc, res.self_bleu)

		if haveref:
			res.ref_bleu = mres.bleu
			res.g2, res.h2 = getmean(res.acc, res.ref_bleu)
			res.overall = res.g2
		else:
			res.overall = res.self_g2
		return res

	with open(args.output, "w") as g:
		metric_names = ["acc", "self_bleu", "ref_bleu", "ppl", "self_g2", "self_h2", "g2", "h2", "overall"]
		# print("\t".join(["domain"] + metric_names))
		g.write("\t".join(["domain"] + metric_names) + "\n")

		for set_name, domain  in product(["dev", "test"], [0, 1]):
			filepath = args[f"{set_name}{domain}"]
			print(f"evaluating {filepath}...")
			if filepath:
				with torch.no_grad():
					res = calc(filepath, set_name, domain)
				output_value = [f"{set_name}{domain}"] + [(("%.3f" % res[key]) if key in res else "n/a") for key in metric_names]
				# print("\t".join(output_value))
				g.write("\t".join([x for x in output_value]) + "\n")

	print(f"output to {args.output}")
	with open(args.output) as g:
		for line in g:
			print(line.strip())

if __name__ == '__main__':
	import sys
	run(*sys.argv[1:])
