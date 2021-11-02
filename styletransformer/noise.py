import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from itertools import chain

# from https://github.com/cindyxinyiwang/deep-latent-sequence-model
class NoiseLayer(nn.Module):
	"""Add noise to words,
	wrapper class of noise function from FAIR (upon some modification):
	https://github.com/facebookresearch/UnsupervisedMT/blob/master/NMT/src/trainer.py
	"""
	def __init__(self, word_blank, word_dropout, word_shuffle, word_repeat, word_add, *,
				 keep_go=True, keep_eos=True, max_length, dm):
		super(NoiseLayer, self).__init__()
		self.blank_prob = word_blank
		self.dropout_prob = word_dropout
		self.shuffle_weight = word_shuffle
		self.repeat_prob = word_repeat
		self.add_prob = word_add

		self.pad_index = dm.pad_id
		self.blank_index = dm.unk_id
		self.eos_index = dm.eos_id

		self.keep_go = keep_go
		self.keep_eos = keep_eos
		self.max_length = max_length
		self.dm = dm

        # prepare for word add
        # heuristic rule: increase the probability for stylist word (from tf-idf)
		def get_prob(domain):
			cnt = np.zeros((dm.frequent_vocab_size,))
			sents = dm.get_all_batch(f"train_{domain}")["sent"]
			for value, num in np.array(np.unique(list(chain(*sents)), return_counts=True)).T:
				cnt[value] = num
			cnt[:4] = 0
			cnt = cnt / np.sum(cnt)

			docnum = np.zeros((dm.frequent_vocab_size,))
			for sent in sents:
				for value in np.unique(sent):
					docnum[value] += 1
			docnum = docnum / len(sents)

			return cnt, docnum
		cnt0, docnum0 = get_prob(0)
		cnt1, docnum1 = get_prob(1)

		idf = np.maximum(np.log(0.3 / (docnum0 + docnum1 + 1e-10)), 0)
		p0 = cnt0 * idf
		p0 = p0 / np.sum(p0)

		p1 = cnt1 * idf
		p1 = p1 / np.sum(p1)

		self.register_buffer("cnt", torch.tensor(np.array([p0, p1]), dtype=torch.float))

	def forward(self, words, lengths, rev_domain):
		"""perform shuffle, dropout, and blank operations,
		note that the input is required to have bos_index at the start and
		eos_index at the end
		Args:
			words (LongTensor): the word ids, (seq_len, batch_size)
			lengths (np.ndarray): (batch_size)
		"""
		lengths = lengths.detach().cpu().numpy()
		words, lengths = self.word_shuffle(words, lengths)
		words, lengths = self.word_blank(words, lengths)
		words, lengths = self.word_add(words, lengths, rev_domain)
		words, lengths = self.word_dropout(words, lengths)
		lengths = torch.tensor(lengths, dtype=torch.long, device=words.device)
		return words, lengths

	def word_add(self, x, l, rev_domain):
		if self.add_prob == 0:
			return x, l
		assert self.add_prob < 1

		mask = np.concatenate(
			[np.ones((1, x.size(1)), dtype=int),
			np.random.rand(self.max_length - 1, x.size(1)) >= self.add_prob], 0)
		repeat_idx = np.cumsum(mask, 0) - 1
		repeat_idx = np.maximum(repeat_idx, np.expand_dims(np.arange(-self.max_length, 0), 1) + np.expand_dims(l, 0))
		eos_mask = repeat_idx >= l
		l_new =  np.sum(~eos_mask, 0)
		for i in range(len(l_new)):
			if mask[l_new[i] - 1, i] == 0 and l_new[i] > 1:
				l_new[i] -= 1
		repeat_idx = np.minimum(repeat_idx, np.expand_dims(l - 1, 0))

		mask = torch.tensor(mask, dtype=torch.long, device=x.device)
		eos_mask = torch.tensor(eos_mask, dtype=torch.long, device=x.device)
		cnt = torch.gather(self.cnt, 0, rev_domain.unsqueeze(-1).expand(-1, self.cnt.shape[1]))
		adding_words = torch.multinomial(cnt, self.max_length, True).transpose(0, 1)
		x_new = x.gather(0, torch.tensor(repeat_idx, dtype=torch.long, device=x.device))
		x_new = x_new * mask + adding_words * (1 - mask)
		x_new = x_new * (1 - eos_mask) + self.dm.eos_id * eos_mask

		return x_new[:max(l_new), :], l_new

	def word_blank(self, x, l):
		"""
		Randomly blank input words.
		Args:
			words (LongTensor): the word ids, (seq_len, batch_size)
			lengths (np.ndarray): (batch_size)
		"""
		if self.blank_prob == 0:
			return x, l
		assert 0 < self.blank_prob <= 1

		keep = np.random.rand(x.size(0), x.size(1)) >= self.blank_prob

		if self.keep_go:
			keep[0] = 1  # do not blank the start sentence symbol
		if self.keep_eos:
			for i in range(len(l)):
				keep[l[i]-1:, i] = 1
		else:
			for i in range(len(l)):
				keep[l[i]:, i] = 1

		keep = torch.tensor(keep, dtype=torch.long, device=x.device)
		# if isinstance(x, (torch.LongTensor, torch.cuda.LongTensor)):
		# 	keep = torch.LongTensor(keep).cuda()
		# else:
		# 	keep = torch.FloatTensor(keep).cuda()

		if len(x.shape) == 3:
			keep = keep.unsqueeze(-1)

		x2 = x * keep + self.blank_index * (1 - keep)
		return x2, l

	def word_dropout(self, x, l):
		"""
		Randomly drop and repeat input words.
		Args:
			words (LongTensor): the word ids, (seq_len, batch_size)
			lengths (LongTensor): (batch_size)
		"""
		if self.dropout_prob == 0 and self.repeat_prob == 0:
			return x, l
		assert self.dropout_prob < 1
		assert self.repeat_prob < 1

		# define words to drop
		# bos_index = self.bos_index[lang_id]
		# assert (x[0] == bos_index).sum() == l.size(0)
		keepmask = np.random.rand(self.max_length, x.size(1)) >= self.dropout_prob
		repeat_idx = np.concatenate(
			[np.zeros((1, x.size(1)), dtype=int),
			np.cumsum(np.random.rand(self.max_length - 1, x.size(1)) >= self.repeat_prob, axis=0)], 0
		)
		# repeat_idx = np.maximum(repeat_idx, \
		# 	np.expand_dims(np.arange(-self.max_length, 0), 1) + np.expand_dims(l, 1))

		# be sure to drop entire words
		# bpe_end = self.bpe_end[lang_id][x]
		# word_idx = bpe_end[::-1].cumsum(0)[::-1]
		# word_idx = word_idx.max(0)[None, :] - word_idx

		assert not self.keep_go
		assert self.keep_eos

		sentences = []
		lengths = []
		for i in range(len(l)):
			if x[l[i] - 1, i] != self.eos_index:
				words = x[:l[i], i].tolist()
			else:
				words = x[:l[i] - 1, i].tolist()
			# randomly drop words from the input
			new_s = [w for j, w in enumerate(words) if keepmask[j, i]]
			# we need to have at least one word in the sentence (more than the start / end sentence symbols)
			if len(new_s) == 0:
				new_s.append(words[np.random.randint(0, len(words))])

			repeat_idx_i = np.maximum(repeat_idx[:, i], np.arange(-self.max_length, 0) + len(new_s))
			new_s2 = [new_s[j] for j in filter(lambda x: x < len(new_s), repeat_idx_i.tolist())]

			if len(new_s2) < self.max_length and new_s2[-1] != self.eos_index:
				new_s2.append(self.eos_index)

			sentences.append(new_s2)
			lengths.append(len(new_s2))
		# re-construct input
		l2 = lengths
		x2 = x.new_full((max(l2), len(l2)), fill_value=self.pad_index)
		for i in range(len(l2)):
			x2[:l2[i], i].copy_(x.new_tensor(sentences[i]))
		return x2, l2

	def word_shuffle(self, x, l):
		"""
		Randomly shuffle input words.
		Args:
			words (LongTensor): the word ids, (seq_len, batch_size)
			lengths (np.ndarray): (batch_size)
		"""
		if self.shuffle_weight == 0:
			return x, l

		# define noise word scores
		noise = np.random.uniform(0, self.shuffle_weight, size=(x.size(0), x.size(1)))

		assert self.shuffle_weight > 1

		perm = []
		for i in range(len(l)):
			s = 1 if self.keep_go else 0
			t = l[i] - 1 if self.keep_eos else l[i]

			# generate a random permutation
			scores = np.arange(t - s) + noise[s:t, i]
			# scores += 1e-6 * np.arange(l[i] - 1)  # ensure no reordering inside a word
			permutation = scores.argsort()
			# shuffle words
			if self.keep_go:
				perm.append(np.concatenate([[0], permutation + 1, np.arange(x.size(0) - t) + t]))
			else:
				perm.append(np.concatenate([permutation, np.arange(x.size(0) - t) + t]))

		perm = torch.tensor(perm, device=x.device, dtype=torch.long).transpose(0, 1)
		if len(x.shape) == 3:
			perm = perm.unsqueeze(-1).expand(-1, -1, x.shape[-1])
		x2 = torch.gather(x, 0, perm)
		return x2, l
