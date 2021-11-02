"""Dataloader for language generation"""
import numpy as np
import random
from collections import Counter, OrderedDict
from itertools import chain
from nltk.tokenize import WordPunctTokenizer
import pickle as pkl
import sys

from cotk.dataloader import LanguageProcessing, GeneralVocab, FieldContext, SentenceDefault

class PredefinedLanguageGeneration(LanguageProcessing):
	def __init__(self, file_id, *,
			max_sent_length=None, \
			convert_to_lower_letter=None, \
			set_name=None, \
			fields=None
			):

		word_list = list(map(lambda x: x.strip(), open(file_id + "/word_list.txt", encoding="utf-8").readlines()))

		vocab = GeneralVocab.from_predefined(["<pad>", "<go>", "<eos>", "<unk>"] + word_list, len(word_list) + 4, \
			OrderedDict([("pad", "<pad>"), ("go", "<go>"), ("eos", "<eos>"), ("unk", "<unk>")]))

		if fields is None:
			if set_name is None:
				fields = OrderedDict([("sent", "SentenceDefault")])
			else:
				fields = {key: OrderedDict([("sent", "SentenceDefault")]) for key in set_name}

		with FieldContext.set_parameters(vocab=vocab,\
				tokenizer="nltk", \
				vocab_from_mappings={
					"train": "train",
					"training": "train",
					"dev": "test",
					"development": "test",
					"valid": "test",
					"validation": "test",
					"test": "test",
					"evaluation": "test",
					"fake": "train",
					"test_adv": "test"
				},
				max_sent_length=max_sent_length,
				convert_to_lower_letter=convert_to_lower_letter):
			super().__init__(file_id, fields)
		self.set_default_field("train", "sent")


class PredefinedSeq2seq(LanguageProcessing):
	def __init__(self, file_id, *,
			max_sent_length=None, \
			convert_to_lower_letter=None, \
			set_name=None, \
			sent0_key="sent0", sent1_key="sent1"
			):

		word_list = list(map(lambda x: x.strip(), open(file_id + "/word_list.txt", encoding="utf-8").readlines()))

		vocab = GeneralVocab.from_predefined(["<pad>", "<go>", "<eos>", "<unk>"] + word_list, len(word_list) + 4, \
			OrderedDict([("pad", "<pad>"), ("go", "<go>"), ("eos", "<eos>"), ("unk", "<unk>")]))

		if set_name is None:
			fields = OrderedDict([(sent0_key, "SentenceDefault"), (sent1_key, "SentenceDefault")])
		else:
			fields = {key: OrderedDict([(sent0_key, "SentenceDefault"), (sent1_key, "SentenceDefault")]) for key in set_name}


		with FieldContext.set_parameters(vocab=vocab,\
				tokenizer="space", \
				vocab_from_mappings={
					"train": "train",
					"training": "train",
					"dev": "test",
					"development": "test",
					"valid": "test",
					"validation": "test",
					"test": "test",
					"evaluation": "test",
					"fake": "train"
				},
				max_sent_length=max_sent_length,
				convert_to_lower_letter=convert_to_lower_letter):
			super().__init__(file_id, fields)
		self.set_default_field("train", sent0_key)

class PredefinedMachineTranslation(LanguageProcessing):
	def __init__(self, file_id, *,
			max_sent_length=None, \
			convert_to_lower_letter=None, \
			set_name=None
			):

		word_list1 = list(map(lambda x: x.strip(), open(file_id + "/word_list_src.txt", encoding="utf-8").readlines()))
		word_list2 = list(map(lambda x: x.strip(), open(file_id + "/word_list_tar.txt", encoding="utf-8").readlines()))

		vocab1 = GeneralVocab.from_predefined(["<pad>", "<go>", "<eos>", "<unk>"] + word_list1, len(word_list1) + 4, \
			OrderedDict([("pad", "<pad>"), ("go", "<go>"), ("eos", "<eos>"), ("unk", "<unk>")]))
		vocab2 = GeneralVocab.from_predefined(["<pad>", "<go>", "<eos>", "<unk>"] + word_list2, len(word_list2) + 4, \
			OrderedDict([("pad", "<pad>"), ("go", "<go>"), ("eos", "<eos>"), ("unk", "<unk>")]))

		vocab_from = {
			"train": "train",
			"training": "train",
			"dev": "test",
			"development": "test",
			"valid": "test",
			"validation": "test",
			"test": "test",
			"evaluation": "test",
			"fake": "train"
		}

		with FieldContext.set_parameters(\
				tokenizer="space", \
		):
			sent1 = SentenceDefault(None, vocab1, vocab_from, max_sent_length, convert_to_lower_letter)
			sent2 = SentenceDefault(None, vocab2, vocab_from, max_sent_length, convert_to_lower_letter)

			if set_name is None:
				fields = OrderedDict([("post", sent1), ("resp", sent2)])
			else:
				fields = {key: OrderedDict([("post", sent1), ("resp", sent2)]) for key in set_name}
			super().__init__(file_id, fields)
		# self.set_default_field("train", "post")

class PredefinedStyleTransfer(LanguageProcessing):
	def __init__(self, file_id, *,
			max_sent_length=None, \
			convert_to_lower_letter=None,
			fields=None
			):

		word_list = list(map(lambda x: x.strip(), open(file_id + "/word_list.txt", encoding="utf-8").readlines()))

		vocab = GeneralVocab.from_predefined(["<pad>", "<go>", "<eos>", "<unk>"] + word_list, len(word_list) + 4, \
			OrderedDict([("pad", "<pad>"), ("go", "<go>"), ("eos", "<eos>"), ("unk", "<unk>")]))

		if fields is None:
			fields = {
				"train_0": OrderedDict([("sent", "SentenceDefault")]),
				"train_1": OrderedDict([("sent", "SentenceDefault")]),
				"dev_0": OrderedDict([("sent", "SentenceDefault")]),
				"dev_1": OrderedDict([("sent", "SentenceDefault")]),
				"test_0": OrderedDict([("sent", "SentenceDefault"), ("ref", "SentenceDefault")]),
				"test_1": OrderedDict([("sent", "SentenceDefault"), ("ref", "SentenceDefault")]),
			}


		vocab_from_mappings = {key: "test" for key in fields.keys() }

		with FieldContext.set_parameters(vocab=vocab,\
				tokenizer="space", \
				vocab_from_mappings=vocab_from_mappings, \
				max_sent_length=max_sent_length,
				convert_to_lower_letter=convert_to_lower_letter):
			super().__init__(file_id, fields)
		self.set_default_field("train_0", "sent")

class PredefinedSeq2seqWithCandidates(LanguageProcessing):
	def __init__(self, file_id, *,
			max_sent_length=None, \
			convert_to_lower_letter=None,
			fields=None
			):

		word_list = list(map(lambda x: x.strip(), open(file_id + "/word_list.txt", encoding="utf-8").readlines()))

		vocab = GeneralVocab.from_predefined(["<pad>", "<go>", "<eos>", "<unk>"] + word_list, len(word_list) + 4, \
			OrderedDict([("pad", "<pad>"), ("go", "<go>"), ("eos", "<eos>"), ("unk", "<unk>")]))

		if fields is None:
			fields = {
				"train": OrderedDict([("post", "SentenceDefault"), ("resp", "SentenceDefault")]),
				"dev": OrderedDict([("post", "SentenceDefault"), ("resp", "SentenceCandidateDefault")]),
				"test": OrderedDict([("post", "SentenceDefault"), ("resp", "SentenceCandidateDefault")]),
			}


		vocab_from_mappings = {key: "test" for key in fields.keys() }

		with FieldContext.set_parameters(vocab=vocab,\
				tokenizer="space", \
				vocab_from_mappings=vocab_from_mappings, \
				max_sent_length=max_sent_length,
				convert_to_lower_letter=convert_to_lower_letter):
			super().__init__(file_id, fields)
		self.set_default_field("train", "post")
