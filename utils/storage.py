# coding:utf-8

class Storage(dict):
	def __init__(self, *args, **kw):
		dict.__init__(self, *args, **kw)

	def __getattr__(self, key):
		if super().__contains__(key):
			return self[key]
		else:
			raise AttributeError("This storage has no attribute %s" % key)

	def __setitem__(self, key, value):
		key = str(key).split("/")
		if len(key) > 1:
			if not super().__contains__(key[0]):
				super().__setitem__(key[0], Storage())
			self[key[0]].__setitem__("/".join(key[1:]), value)
		else:
			super().__setitem__(key[0], value)

	def __contains__(self, key):
		try:
			self.__getitem__(key)
			return True
		except KeyError:
			return False

	def __getitem__(self, key):
		key = str(key).split("/")
		if len(key) > 1:
			if not super().__contains__(key[0]):
				raise KeyError("no such key %s in storage" % key[0])
			return self[key[0]].__getitem__(".".join(key[1:]))
		else:
			return super().__getitem__(key[0])

	def __setattr__(self, key, value):
		self[key] = value

	def __delattr__(self, key):
		del self[key]

	def __sub__(self, b):
		'''Delete all items which b has (ignore values).
		'''
		res = Storage()
		for i, j in self.items():
			if i not in b:
				res[i] = j
			elif isinstance(j, Storage) and isinstance(b[i], Storage):
				diff = j - b[i]
				res[i] = diff
		return res

	def __xor__(self, b):
		'''Return different items in two Storages (only intersection keys).
		'''
		res = Storage()
		for i, j in self.items():
			if i in b:
				if isinstance(j, Storage) and isinstance(b[i], Storage):
					res[i] = j ^ b[i]
				elif j != b[i]:
					res[i] = (j, b[i])
		return res

	def update(self, b):
		'''will NOT overwrite existed key.
		'''
		for i, j in b.items():
			if i not in self:
				self[i] = j
			elif isinstance(self[i], Storage) and isinstance(j, Storage):
				self[i].update(j)

	def listitems(self, prefix="", split="/"):
		for key, value in self.items():
			if isinstance(value, Storage):
				for k, v in value.listitems(prefix=prefix + key + split, split=split):
					yield k, v
			else:
				yield prefix + key, value
