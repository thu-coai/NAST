from torch.optim import Optimizer

class ReduceLROnLambda():
	def __init__(self, optimizer, func, factor=0.1, patience=5, \
				 verbose=False, min_lr=0, eps=1e-8):
		if factor >= 1.0:
			raise ValueError('Factor should be < 1.0.')
		self.factor = factor

		if not isinstance(optimizer, Optimizer):
			raise TypeError('{} is not an Optimizer'.format(\
				type(optimizer).__name__))
		self.optimizer = optimizer

		if isinstance(min_lr, list) or isinstance(min_lr, tuple):
			if len(min_lr) != len(optimizer.param_groups):
				raise ValueError("expected {} min_lrs, got {}".format(\
					len(optimizer.param_groups), len(min_lr)))
			self.min_lrs = list(min_lr)
		else:
			self.min_lrs = [min_lr] * len(optimizer.param_groups)

		self.func = func
		self.verbose = verbose
		self.eps = eps
		self.patience = patience
		self.history_data = None
		self.now_patience = 0
		self.now_lr = [param_group["lr"] for param_group in self.optimizer.param_groups]

	def step(self, metrics):
		flag, self.history_data = self.func(metrics, self.history_data)
		if flag:
			self.now_patience += 1
			if self.verbose:
				print("No improvement. now patience: %d" % self.now_patience)
			if self.now_patience >= self.patience:
				self._reduce_lr()
				self.now_patience = 0
		else:
			if self.verbose:
				print("Improvement! now patience clear")
			self.now_patience = 0

	def _reduce_lr(self):
		for i, param_group in enumerate(self.optimizer.param_groups):
			old_lr = float(param_group['lr'])
			new_lr = max(old_lr * self.factor, self.min_lrs[i])
			if old_lr - new_lr > self.eps:
				param_group['lr'] = new_lr
				self.now_lr[i] = new_lr
				if self.verbose:
					print('Reducing learning rate' \
						  ' of group {} to {:.4e}.'.format(i, new_lr))

	def state_dict(self):
		return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'func'}}

	def load_state_dict(self, state_dict):
		self.__dict__.update(state_dict)
		if "now_lr" in state_dict:
			for i, param_group in enumerate(self.optimizer.param_groups):
				param_group['lr'] = state_dict["now_lr"][i]
