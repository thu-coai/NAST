# coding:utf-8

def run(*argv):
	import argparse
	import time

	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
	from utils import Storage

	parser = argparse.ArgumentParser(description='Train a language model')
	args = Storage()

	parser.add_argument('--name', type=str, default=None,
		help='The name of your model, used for tensorboard, etc. Default: runXXXXXX_XXXXXX (initialized by current time)')
	parser.add_argument('--restore', type=str, default=None,
		help='Checkpoints name to load. \
			"NAME_last" for the last checkpoint of model named NAME. "NAME_best" means the best checkpoint. \
			You can also use "last" and "best", by default use last model you run. \
			It can also be an url started with "http". \
			Attention: "NAME_last" and "NAME_best" are not guaranteed to work when 2 models with same name run in the same time. \
			"last" and "best" are not guaranteed to work when 2 models run in the same time.\
			Default: None (don\'t load anything)')
	parser.add_argument('--mode', type=str, default="train",
		help='"train" or "test". Default: train')

	parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate. Default: 1e-5')
	parser.add_argument('--batch_size', type=int, default=16, help="batch size")

	parser.add_argument('--gpt2', type=str, default="gpt2", help="the name of pretrained model")
	parser.add_argument('--droprate', type=int, default=0.5, help="dropout rate")

	parser.add_argument('--dataid', type=str, default='../data/yelp', help='the data path')
	parser.add_argument('--epoch', type=int, default=10, help="Epoch for training. Default: 10")
	parser.add_argument('--batch_per_epoch', type=int, default=1000, help="Batches per epoch. Default: 1000")

	parser.add_argument('--out_dir', type=str, default="./output",
		help='Output directory for test output. Default: ./output')
	parser.add_argument('--log_dir', type=str, default="./tensorboard",
		help='Log directory for tensorboard. Default: ./tensorboard')
	parser.add_argument('--model_dir', type=str, default="./model",
		help='Checkpoints directory for model. Default: ./model')
	parser.add_argument('--cache_dir', type=str, default="./cache",
		help='Checkpoints directory for cache. Default: ./cache')
	parser.add_argument('--cuda', action="store_true", help='Use cuda.')
	parser.add_argument('--debug', action='store_true', help='Enter debug mode (using ptvsd).')
	parser.add_argument('--cache', action='store_true', help='Use cache for speeding up load data.')
	parser.add_argument('--seed', type=int, default=0, help='Specify random seed. Default: 0')

	parser.add_argument('--dryrun', action='store_true')

	cargs = parser.parse_args(argv)

	# Editing following arguments to bypass command line.
	args.name = cargs.name or time.strftime("run%Y%m%d_%H%M%S", time.localtime())

	args.restore = cargs.restore
	args.mode = cargs.mode
	args.datapath = cargs.dataid
	args.epochs = cargs.epoch
	args.out_dir = cargs.out_dir
	args.log_dir = cargs.log_dir
	args.model_dir = cargs.model_dir
	args.cache_dir = cargs.cache_dir
	args.debug = cargs.debug
	args.cache = cargs.cache
	args.cuda = "cuda" if cargs.cuda else "cpu"

	# The following arguments are not controlled by command line.
	args.restore_optimizer = False
	#args.restore_other_weights = True
	load_exclude_set = []
	restoreCallback = None

	args.embed_size = 300

	args.premodel = "lm"
	args.gpt2 = cargs.gpt2
	args.droprate = cargs.droprate

	args.batch_per_epoch = cargs.batch_per_epoch
	args.lr = cargs.lr
	args.batch_size = cargs.batch_size
	args.grad_clip = 5
	args.show_sample = [0]  # show which batch when evaluating at tensorboard
	args.max_length = 50
	args.checkpoint_steps = 20
	args.checkpoint_max_to_keep = 5

	args.seed = cargs.seed

	if cargs.dryrun:
		return args

	import random
	random.seed(cargs.seed)
	import torch
	torch.manual_seed(cargs.seed)
	import numpy as np
	np.random.seed(cargs.seed)

	from main import main
	main(args, load_exclude_set, restoreCallback)

if __name__ == '__main__':
	import sys
	run(*sys.argv[1:])
