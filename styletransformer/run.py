def run(*argv):
	import argparse
	import time

	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
	from utils import Storage

	parser = argparse.ArgumentParser()
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

	parser.add_argument('--dataid', type=str, default='../data/yelp', help='the data path')

	parser.add_argument('--out_dir', type=str, default="./output",
		help='Output directory for test output. Default: ./output')
	parser.add_argument('--log_dir', type=str, default="./tensorboard",
		help='Log directory for tensorboard. Default: ./tensorboard')
	parser.add_argument('--model_dir', type=str, default="./model",
		help='Checkpoints directory for model. Default: ./model')
	parser.add_argument('--cache_dir', type=str, default="./cache",
		help='Checkpoints directory for cache. Default: ./cache')
	parser.add_argument('--cpu', action="store_true", help='Use cpu. (use gpu by default)')
	parser.add_argument('--debug', action='store_true', help='Enter debug mode (using ptvsd).')
	parser.add_argument('--cache', action='store_true', help='Use cache for speeding up load data.')
	parser.add_argument('--seed', type=int, default=0, help='Specify random seed. Default: 0')

	parser.add_argument('--epoch', type=int, default=2000, help="Epoch for training. Default: 2000")
	parser.add_argument('--batch_per_epoch', type=int, default=50, help="Batches per epoch. Default: 50")

	parser.add_argument('--pretrain_batch', type=int, default=500, help="update steps for pretrain")

	parser.add_argument('--dis_approx_mode', type=str, default="gumbel") # or "soft"
	parser.add_argument('--gen_approx_mode', type=str, default="gumbel") # or "soft"

	parser.add_argument('--fw_adv_factor', type=float, default=1.5)
	parser.add_argument('--bw_adv_factor', type=float, default=1)

	parser.add_argument('--decoder_input', type=str, default="encoded") # or "raw"

	parser.add_argument('--self_factor', type=float, default=0.5)
	parser.add_argument("--cyc_factor", type=float, default=0.5)
	parser.add_argument("--radam", action="store_true")
	parser.add_argument("--clsrestore", type=str, default="cls_yelp_best", help="the model name for the classifier")
	parser.add_argument("--discriminator_method", type=str, default="Cond") # or "Multi"

	parser.add_argument("--use_learnable", action="store_true")
	parser.add_argument("--length_factor", type=float, default=1)
	parser.add_argument("--z_decode_mode", type=str, default="max")
	parser.add_argument("--pre_z_batch", type=int, default=0)

	cargs = parser.parse_args(argv)

	args.name = cargs.name or time.strftime("run%Y%m%d_%H%M%S", time.localtime())

	args.restore = cargs.restore
	args.mode = cargs.mode
	#args.epochs = cargs.epoch
	args.datapath = cargs.dataid
	args.out_dir = cargs.out_dir
	args.model_dir = cargs.model_dir
	args.cache_dir = cargs.cache_dir
	args.debug = cargs.debug
	args.cache = cargs.cache
	args.cuda = "cpu" if cargs.cpu else "cuda"
	args.restore_optimizer = True

	args.checkpoint_steps = 5
	args.checkpoint_max_to_keep = 20
	args.show_sample = [0]  # show which batch when evaluating at tensorboard

	args.premodel = ""
	args.clsrestore = cargs.clsrestore
	args.epochs = cargs.epoch

	args.use_learnable = cargs.use_learnable

	# the arguments for noise layer
	if args.use_learnable:
		args.word_blank = 0.1
		args.word_dropout = 0.05
		args.word_repeat = 0
		args.word_add = 1 / 0.95 - 1
		args.word_shuffle = 0
	else:
		args.word_blank = 0.2
		args.word_dropout = 0
		args.word_repeat = 0
		args.word_add = 0
		args.word_shuffle = 0

	args.dis_approx_mode = cargs.dis_approx_mode
	args.gen_approx_mode = cargs.gen_approx_mode

	args.fw_adv_factor = cargs.fw_adv_factor
	args.bw_adv_factor = cargs.bw_adv_factor

	args.decoder_input = cargs.decoder_input
	args.radam = cargs.radam

	args.length_factor = cargs.length_factor
	args.z_decode_mode = cargs.z_decode_mode
	args.pre_z_batch = cargs.pre_z_batch

	args.log_dir = cargs.log_dir
	args.save_path = cargs.model_dir
	args.device = args.cuda

	args.discriminator_method = cargs.discriminator_method # 'Multi' or 'Cond'
	args.min_freq = 3
	args.max_length = 16
	args.embed_size = 256
	args.d_model = 256
	args.h = 4
	args.num_styles = 2
	args.num_classes = args.num_styles + 1 if args.discriminator_method == 'Multi' else 2
	args.num_layers = 4
	args.batch_size = 64
	args.lr_F = 0.0001
	args.lr_D = 0.0001
	args.L2 = 0
	args.iter_D = 10
	args.iter_F = 5
	args.F_pretrain_iter = cargs.pretrain_batch
	args.log_steps = 5
	args.eval_steps = cargs.batch_per_epoch
	args.learned_pos_embed = True
	args.dropout = 0
	args.temperature_config = [(1, 0)]  # can be used for decoding temperate annealing, but we fix it as 1 here

	args.slf_factor = cargs.self_factor
	args.cyc_factor = cargs.cyc_factor
	args.inp_drop_prob = 0

	args.seed = cargs.seed

	import random
	random.seed(cargs.seed)
	import torch
	torch.manual_seed(cargs.seed)
	import numpy as np
	np.random.seed(cargs.seed)

	from main import main

	main(args)

if __name__ == '__main__':
	import sys
	run(*sys.argv[1:])
