# coding:utf-8
import logging
import json
from collections import OrderedDict

from cotk.wordvector import WordVector, Glove
from utils.cotk_private.dataloader.predefined_language_generation import PredefinedStyleTransfer, PredefinedLanguageGeneration

from utils import debug, try_cache, cuda_init, Storage
from classifier import Classifier
from languagemodel import LanguageModel

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

	if args.premodel == "classifier":
		data_class = PredefinedLanguageGeneration
		data_arg = Storage()
		data_arg.file_id = args.datapath + "/classifier"
		data_arg.max_sent_length = args.max_sent_length + 1
		data_arg.fields = {
					"train": OrderedDict([("sent", "SentenceDefault"), ("domain", "DenseLabel"), ("adv", "DenseLabel")]),
					"test": OrderedDict([("sent", "SentenceDefault"), ("domain", "DenseLabel"), ("adv", "DenseLabel")]),
				}
	elif args.premodel == "lm":
		data_class = PredefinedLanguageGeneration
		data_arg = Storage()
		data_arg.file_id = args.datapath + "/languagemodel"
		data_arg.max_sent_length = args.max_length + 1
		data_arg.fields = {
			"train": OrderedDict([("sent", "SentenceDefault")]),
			"test": OrderedDict([("sent", "SentenceDefault"), ("domain", "DenseLabel"), ("adv", "DenseLabel")]),
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

	if args.premodel == "classifier":
		model = Classifier(param)
		if args.mode == "train":
			model.train_process()
		elif args.mode == "test":
			test_res = model.test_process()
		else:
			raise ValueError("Unknown mode")
	elif args.premodel == "lm":
		model = LanguageModel(param)
		if args.mode == "train":
			model.train_process()
		elif args.mode == "test":
			test_res = model.test_process()
		else:
			raise ValueError("Unknown mode")
	else:
		raise ValueError("Unknown premodel")
