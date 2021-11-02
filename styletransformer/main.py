# coding:utf-8
import os
import logging
import json
from collections import Counter, OrderedDict

from utils.cotk_private.dataloader.predefined_language_generation import PredefinedStyleTransfer
from utils import debug, try_cache, cuda_init, Storage
from seq2seq import Seq2seq
from classifier import Classifier

import copy

def main(args, load_exclude_set=None, restoreCallback=None):
	logging.basicConfig(\
		filename=0,\
		level=logging.DEBUG,\
		format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',\
		datefmt='%H:%M:%S')

	if args.debug:
		debug()
	logging.info(json.dumps(args, indent=2))

	cuda_init(0, args.cuda)

	volatile = Storage()
	volatile.load_exclude_set = load_exclude_set or []
	volatile.restoreCallback = restoreCallback

	data_class = PredefinedStyleTransfer
	data_arg = Storage()
	data_arg.file_id = args.datapath
	data_arg.max_sent_length = args.max_length
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
			args.cache_dir, f"{data_class.__name__}_{args.premodel}")
	else:
		dm = load_dataset(data_arg)

	volatile.dm = dm

	param = Storage()
	param.args = args
	param.volatile = volatile

	model = Seq2seq(param)

	import run_cls
	cls_param = Storage()
	cls_param.args = run_cls.run("--dryrun", "--restore", args.clsrestore)
	cls_param.volatile = copy.copy(volatile)

	classifier = Classifier(cls_param)
	param.volatile.cls = classifier

	if args.mode == "train":
		model.train_process()
	elif args.mode == "test":
		model.test_process()
	else:
		raise ValueError("Unknown mode")
