import math
import torch
from torch import nn
import torch.nn.functional as F
from utils import BaseNetwork
from noise import NoiseLayer
import numpy as np

class Network(BaseNetwork):
	def __init__(self, param):
		super().__init__(param)
		args = self.param.args
		self.model_F = StyleTransformer(param)
		self.model_D = Discriminator(param)
		self.noiseLayer = NoiseLayer(args.word_blank, args.word_dropout, args.word_shuffle, args.word_repeat, args.word_add,
			keep_go=False, max_length=args.max_length, dm=param.volatile.dm)


class StyleTransformer(nn.Module):
	def __init__(self, param):
		super(StyleTransformer, self).__init__()

		self.param = param
		config = args = self.args = param.args
		dm = param.volatile.dm

		num_styles, num_layers = config.num_styles, config.num_layers
		d_model, max_length = config.d_model, config.max_length
		h, dropout = config.h, config.dropout
		learned_pos_embed = config.learned_pos_embed

		self.max_length = config.max_length
		self.eos_idx = dm.eos_id
		self.pad_idx = dm.pad_id
		self.style_embed = Embedding(num_styles, d_model)
		self.embed = EmbeddingLayer(
			dm, d_model, max_length,
			self.pad_idx,
			learned_pos_embed
		)
		self.sos_token = nn.Parameter(torch.randn(d_model))
		self.encoder = Encoder(num_layers, d_model, dm.frequent_vocab_size, h, dropout)
		self.decoder = Decoder(num_layers, d_model, dm.frequent_vocab_size, h, dropout)

		if args.use_learnable:
			self.length_predictor = LengthPredictor(param)

	def forward(self, inp_tokens, inp_lengths, gold_tokens, gold_lengths, style,
				generate=False, differentiable_decode=False, temperature=1.0, *, use_inference_z):
		batch_size = inp_tokens.size(0)
		max_enc_len = inp_tokens.size(1)

		assert torch.max(inp_lengths) <= inp_tokens.shape[1]
		assert max_enc_len <= self.max_length

		pos_idx = torch.arange(self.max_length + 1).unsqueeze(0).expand((batch_size, -1))
		pos_idx = pos_idx.to(inp_lengths.device)

		src_mask = pos_idx[:, :max_enc_len] >= inp_lengths.unsqueeze(-1)
		src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), 1)
		src_mask = src_mask.view(batch_size, 1, 1, max_enc_len + 1)

		style_emb = self.style_embed(style).unsqueeze(1)

		enc_input = torch.cat((style_emb, self.embed(inp_tokens, pos_idx[:, :max_enc_len])), 1)
		memory = self.encoder(enc_input, src_mask) # batch * len * hidden_size

		#sos_token = self.sos_token.view(1, 1, -1).expand(batch_size, -1, -1)
		if self.args.decoder_input == "encoded":
			dec_input_emb = memory
		elif self.args.decoder_input == "raw":
			dec_input_emb = enc_input
		else:
			raise NotImplementedError()

		if self.args.use_learnable:
			if generate:
				dec_input_emb, tgt_lengths, allpos = self.length_predictor(dec_input_emb, inp_lengths, embed=self.embed, use_inference_z=use_inference_z)
				length_loss = None
				gen_error = None
			else:
				dec_input_emb, tgt_lengths, length_loss, gen_error = \
					self.length_predictor(dec_input_emb, inp_lengths, inp_tokens, gold_tokens, gold_lengths, embed=self.embed, use_inference_z=use_inference_z)
				allpos = None

			max_dec_len = dec_input_emb.size(1) - 1
			tgt_mask = pos_idx[:, :max_dec_len] >= tgt_lengths.unsqueeze(-1)
			tgt_mask = torch.cat((torch.zeros_like(tgt_mask[:, :1]), tgt_mask), 1)
			tgt_mask = tgt_mask.view(batch_size, 1, 1, max_dec_len + 1)
		else:
			tgt_mask = src_mask
			tgt_lengths = None
			length_loss = None
			gen_error = None
			allpos = None

		log_probs = self.decoder(
			dec_input_emb, memory,
			src_mask, tgt_mask,
			temperature
		)

		return log_probs[:, 1:], tgt_lengths, length_loss, gen_error, allpos

class Discriminator(nn.Module):
	def __init__(self, param):
		super(Discriminator, self).__init__()

		self.param = param
		self.args = args = config = param.args
		dm = self.param.volatile.dm

		num_styles, num_layers = config.num_styles, config.num_layers
		d_model, max_length = config.d_model, config.max_length
		h, dropout = config.h, config.dropout
		learned_pos_embed = config.learned_pos_embed
		num_classes = config.num_classes

		self.pad_idx = dm.pad_id
		self.style_embed = Embedding(num_styles, d_model)
		self.embed = EmbeddingLayer(
			dm, d_model, max_length,
			self.pad_idx,
			learned_pos_embed
		)
		self.cls_token = nn.Parameter(torch.randn(d_model))
		self.encoder = Encoder(num_layers, d_model, dm.frequent_vocab_size, h, dropout)
		self.classifier = Linear(d_model, num_classes)

	def forward(self, inp_tokens, inp_lengths, style=None):
		batch_size = inp_tokens.size(0)
		num_extra_token = 1 if style is None else 2
		max_seq_len = inp_tokens.size(1)

		pos_idx = torch.arange(max_seq_len).unsqueeze(0).expand((batch_size, -1)).to(inp_lengths.device)
		mask = pos_idx >= inp_lengths.unsqueeze(-1)
		for _ in range(num_extra_token):
			mask = torch.cat((torch.zeros_like(mask[:, :1]), mask), 1)
		mask = mask.view(batch_size, 1, 1, max_seq_len + num_extra_token)

		cls_token = self.cls_token.view(1, 1, -1).expand(batch_size, -1, -1)

		enc_input = cls_token
		if style is not None:
			style_emb = self.style_embed(style).unsqueeze(1)
			enc_input = torch.cat((enc_input, style_emb), 1)

		enc_input = torch.cat((enc_input, self.embed(inp_tokens, pos_idx)), 1)

		encoded_features = self.encoder(enc_input, mask)
		logits = self.classifier(encoded_features[:, 0])

		return F.log_softmax(logits, -1)



class Encoder(nn.Module):
	def __init__(self, num_layers, d_model, vocab_size, h, dropout):
		super(Encoder, self).__init__()
		self.layers = nn.ModuleList([EncoderLayer(d_model, h, dropout) for _ in range(num_layers)])
		self.norm = LayerNorm(d_model)

	def forward(self, x, mask):
		y = x

		assert y.size(1) == mask.size(-1)

		for layer in self.layers:
			y = layer(y, mask)

		return self.norm(y)

class Decoder(nn.Module):
	def __init__(self, num_layers, d_model, vocab_size, h, dropout):
		super(Decoder, self).__init__()
		self.layers = nn.ModuleList([DecoderLayer(d_model, h, dropout) for _ in range(num_layers)])
		self.norm = LayerNorm(d_model)
		self.generator = Generator(d_model, vocab_size)

	def forward(self, x, memory, src_mask, tgt_mask, temperature):
		y = x

		assert y.size(1) == tgt_mask.size(-1)

		for layer in self.layers:
			y = layer(y, memory, src_mask, tgt_mask)

		return self.generator(self.norm(y), temperature)
	def incremental_forward(self, x, memory, src_mask, tgt_mask, temperature, prev_states=None):
		y = x

		new_states = []


		for i, layer in enumerate(self.layers):
			y, new_sub_states = layer.incremental_forward(
				y, memory, src_mask, tgt_mask,
				prev_states[i] if prev_states else None
			)

			new_states.append(new_sub_states)

		new_states.append(torch.cat((prev_states[-1], y), 1) if prev_states else y)
		y = self.norm(new_states[-1])[:, -1:]

		return self.generator(y, temperature), new_states

class Generator(nn.Module):
	def __init__(self, d_model, vocab_size):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab_size)

	def forward(self, x, temperature):
		return F.log_softmax(self.proj(x) / temperature, dim=-1)

class EmbeddingLayer(nn.Module):
	def __init__(self, dm, d_model, max_length, pad_idx, learned_pos_embed, wordvec=None):
		super(EmbeddingLayer, self).__init__()
		self.token_embed = Embedding(dm.frequent_vocab_size, d_model)
		self.pos_embed = Embedding(max_length, d_model)
		self.vocab_size = dm.frequent_vocab_size
		if wordvec is not None:
			self.token_embed = nn.Embedding.from_pretrained(torch.tensor(wordvec, dtype=torch.float32))

	def forward(self, x, pos):
		if len(x.size()) == 2:
			y = self.token_embed(x) + self.pos_embed(pos)
		else:
			y = torch.matmul(x, self.token_embed.weight) + self.pos_embed(pos)

		return y

class EncoderLayer(nn.Module):
	def __init__(self, d_model, h, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = MultiHeadAttention(d_model, h, dropout)
		self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
		self.sublayer =  nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

	def forward(self, x, mask):
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.pw_ffn)


class DecoderLayer(nn.Module):
	def __init__(self, d_model, h, dropout):
		super(DecoderLayer, self).__init__()
		self.self_attn = MultiHeadAttention(d_model, h, dropout)
		self.src_attn = MultiHeadAttention(d_model, h, dropout)
		self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
		self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

	def forward(self, x, memory, src_mask, tgt_mask):
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		return self.sublayer[2](x, self.pw_ffn)

	def incremental_forward(self, x, memory, src_mask, tgt_mask, prev_states=None):
		new_states = []
		m = memory

		x = torch.cat((prev_states[0], x), 1) if prev_states else x
		new_states.append(x)
		x = self.sublayer[0].incremental_forward(x, lambda x: self.self_attn(x[:, -1:], x, x, tgt_mask))
		x = torch.cat((prev_states[1], x), 1) if prev_states else x
		new_states.append(x)
		x = self.sublayer[1].incremental_forward(x, lambda x: self.src_attn(x[:, -1:], m, m, src_mask))
		x = torch.cat((prev_states[2], x), 1) if prev_states else x
		new_states.append(x)
		x = self.sublayer[2].incremental_forward(x, lambda x: self.pw_ffn(x[:, -1:]))
		return x, new_states

class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, h, dropout):
		super(MultiHeadAttention, self).__init__()
		assert d_model % h == 0
		self.d_k = d_model // h
		self.h = h
		self.head_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
		self.fc = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, query, key, value, mask):
		batch_size = query.size(0)

		query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
							 for x, l in zip((query, key, value), self.head_projs)]

		attn_feature, _ = scaled_attention(query, key, value, mask)

		attn_concated = attn_feature.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

		return self.fc(attn_concated)

def scaled_attention(query, key, value, mask):
	d_k = query.size(-1)
	scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
	scores.masked_fill_(mask, float('-inf'))
	attn_weight = F.softmax(scores, -1)
	attn_feature = attn_weight.matmul(value)

	return attn_feature, attn_weight

class PositionwiseFeedForward(nn.Module):
	def __init__(self, d_model, dropout):
		super(PositionwiseFeedForward, self).__init__()
		self.mlp = nn.Sequential(
			Linear(d_model, 4 * d_model),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			Linear(4 * d_model, d_model),
		)

	def forward(self, x):
		return self.mlp(x)

class SublayerConnection(nn.Module):
	def __init__(self, d_model, dropout):
		super(SublayerConnection, self).__init__()
		self.layer_norm = LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		y = sublayer(self.layer_norm(x))
		return x + self.dropout(y)

	def incremental_forward(self, x, sublayer):
		y = sublayer(self.layer_norm(x))
		return x[:, -1:] + self.dropout(y)

def Linear(in_features, out_features, bias=True, uniform=True):
	m = nn.Linear(in_features, out_features, bias)
	if uniform:
		nn.init.xavier_uniform_(m.weight)
	else:
		nn.init.xavier_normal_(m.weight)
	if bias:
		nn.init.constant_(m.bias, 0.)
	return m

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
	m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
	nn.init.xavier_uniform_(m.weight)
	nn.init.constant_(m.weight[padding_idx], 0)
	return m

def LayerNorm(embedding_dim, eps=1e-6):
	m = nn.LayerNorm(embedding_dim, eps)
	return m


class HiddenDecoder(nn.Module):
	def __init__(self, num_layers, d_model, h, dropout):
		super(HiddenDecoder, self).__init__()
		self.layers = nn.ModuleList([DecoderLayer(d_model, h, dropout) for _ in range(num_layers)])
		self.norm = LayerNorm(d_model)

	def forward(self, x, memory, src_mask, tgt_mask):
		y = x
		assert y.size(1) == tgt_mask.size(-1)
		for layer in self.layers:
			y = layer(y, memory, src_mask, tgt_mask)
		return y

	def incremental_forward(self, x, memory, src_mask, tgt_mask, prev_states=None):
		y = x
		new_states = []
		for i, layer in enumerate(self.layers):
			y, new_sub_states = layer.incremental_forward(
				y, memory, src_mask, tgt_mask,
				prev_states[i] if prev_states else None
			)
			new_states.append(new_sub_states)
		new_states.append(torch.cat((prev_states[-1], y), 1) if prev_states else y)
		y = self.norm(new_states[-1])[:, -1:]
		return y, new_states

class LengthPredictor(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.param = param
		self.args = args = param.args

		self.pos_embed = Embedding(args.max_length + 1, args.d_model, padding_idx=0)
		self.decoder = HiddenDecoder(3, args.d_model, args.h, args.dropout)

		self.max_skip = 4
		self.sqrt_d_model = np.sqrt(self.args.d_model)
		self.linear = nn.Linear(args.d_model, self.max_skip)
		self.wq = nn.Linear(args.d_model, args.d_model)
		self.wk = nn.Linear(args.d_model, args.d_model)

	def match(self, origin_token, origin_length, target_token, target_length, embed):
		unk_id = self.param.volatile.dm.unk_id
		origin_token = origin_token.clone()
		if len(origin_token.shape) == 2:
			origin_token[origin_token == unk_id] = 0
			origin_emb = embed.token_embed(origin_token)
		else:
			origin_token[:, :, 0] += origin_token[:, :, unk_id]
			origin_token[:, :, unk_id] *= 0
			origin_emb = torch.matmul(origin_token, embed.token_embed.weight)

		target_emb = embed.token_embed(target_token)

		tar_seqlen = target_token.shape[1]
		batch_size = target_token.shape[0]
		src_seqlen = origin_token.shape[1]

		cost0 = (target_emb ** 2).sum(dim=-1) + 1e-6 # batch * tar_seqlen
		cost = torch.cdist(target_emb, origin_emb) ** 2 # batch * tar_seqlen * src_seqlen
		src_mask = torch.arange(src_seqlen, device=cost.device).unsqueeze(0).expand((batch_size, -1)) >= origin_length.unsqueeze(-1) # batch * src_seqlen
		cost = cost.masked_fill(src_mask.unsqueeze(1), 1e9)
		f = torch.cat([cost[:, 0, :self.max_skip - 1], torch.ones_like(cost[:, 0, self.max_skip - 1:]) * 1e9], dim=1) # batch * src_seqlen
		src_mask = src_mask.float()

		padding_f = torch.ones(batch_size, self.max_skip, dtype=torch.float32, device=f.device) * 1e9
		tar_mask = (torch.arange(tar_seqlen, device=f.device).unsqueeze(0).expand((batch_size, -1)) >= target_length.unsqueeze(-1)).float()

		pos = [] # delta

		for i in range(1, tar_seqlen):
			f_padded = torch.cat([padding_f, f], dim=1)
			f_padded = torch.stack([f_padded[:, self.max_skip - j: f_padded.shape[1]-j] for j in range(self.max_skip)], -1) # batch * ori_seqlen * 4
			cost_now = torch.cat([cost0[:, i:i+1].expand(-1, f_padded.shape[1]).unsqueeze(-1), cost[:, i].unsqueeze(-1).expand(-1, -1, self.max_skip - 1)], -1)
			newf, newpos = (f_padded + cost_now).min(dim=-1)
			#f = (newf + cost[:, i]) * (1 - tar_mask[:, i:i+1]) + f * tar_mask[:, i:i+1]
			f = newf * (1 - tar_mask[:, i:i+1]) + f * tar_mask[:, i:i+1]
			newpos = newpos * (1 - tar_mask[:, i:i+1]).long()
			pos.append(newpos)

		last_pos_mask = torch.arange(src_seqlen, device=cost.device).unsqueeze(0).expand((batch_size, -1)) >= origin_length.unsqueeze(-1) - self.max_skip + 1
		f = f.masked_fill(~last_pos_mask, 1e9)
		last_pos = f.min(dim=-1)[1] # batch
		back_track = [last_pos]

		pos = torch.stack(pos, dim=1) # batch * tar_seqlen-1 * ori_seqlen
		for i in range(tar_seqlen-1, 0, -1):
			delta = torch.gather(pos[:, i-1], 1, last_pos.unsqueeze(-1))[:, 0]
			last_pos = last_pos - delta
			back_track.append(last_pos)

		allpos = torch.stack(list(reversed(back_track)), dim=-1) # batch * tar_seqlen
		allpos = allpos * (1-tar_mask).long() + origin_length.unsqueeze(-1) * tar_mask.long()

		# add style token
		allpos = torch.cat([torch.zeros_like(allpos[:, :1]), allpos + 1], dim=1) # batch * (tar_seqlen+1)
		return allpos


	def forward(self, origin_input, origin_length, origin_token=None, target_token=None, target_length=None, embed=None, *, use_inference_z):

		batch_size = origin_input.shape[0]
		origin_seqlen = origin_input.shape[1] - 1

		pos_idx = torch.arange(self.args.max_length + 1, device=origin_input.device).unsqueeze(0).expand((batch_size, -1))

		src_mask = (pos_idx[:, :origin_seqlen] >= origin_length.unsqueeze(-1))
		src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), 1)
		src_mask = src_mask.view(batch_size, 1, 1, origin_seqlen + 1)

		tgt_mask = torch.ones((self.args.max_length + 1, self.args.max_length + 1), device=src_mask.device)
		tgt_mask = (tgt_mask.tril() == 0).view(1, 1, self.args.max_length + 1, self.args.max_length + 1)

		src_mask_float = src_mask.float()[:, 0, 0, :].unsqueeze(-1) # batch * (origin_seqlen + 1) * 1

		eos_embed = embed.token_embed.weight[self.param.volatile.dm.eos_id]
		unk_embed = embed.token_embed.weight[self.param.volatile.dm.unk_id]

		candidates_input = origin_input * (1 - src_mask_float) + eos_embed * src_mask_float
		candidates_input = torch.cat([candidates_input, \
			eos_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, self.max_skip + 1, -1)], dim=1)
		memory = origin_input

		candidates_output = torch.stack( [unk_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, origin_seqlen + 2, -1)] + \
			[candidates_input[:, i:origin_seqlen + 2 + i] for i in range(1, self.max_skip)], dim=-2) # batch * (origin_seqlen + 2) * max_skip * dimension

		if target_token is not None:
			if not use_inference_z:
				assert (origin_length == target_length).sum() == batch_size
				return origin_input, target_length, \
					torch.zeros(1, dtype=torch.float, device=origin_input.device)[0], \
					torch.zeros(1, dtype=torch.float, device=origin_input.device)[0]

			with torch.no_grad():
				allpos = self.match(origin_token, origin_length, target_token, target_length, embed)
				unk_mask = ((allpos - torch.cat([torch.ones_like(allpos[:, :1]) * -1, allpos[:, :-1]], dim=1)) == 0).float().unsqueeze(-1)
			target_seqlen = target_token.shape[1]
			dec_input = torch.gather(candidates_input, 1, allpos.unsqueeze(-1).expand(-1, -1, candidates_input.shape[-1])) * (1 - unk_mask) +\
						unk_embed.unsqueeze(0).unsqueeze(0).expand(allpos.shape[0], allpos.shape[1], -1) * unk_mask

			dec_input_new = dec_input + self.pos_embed(pos_idx[:, :target_seqlen + 1])

			hidden = self.decoder(
				dec_input_new, memory,
				src_mask, tgt_mask[:, :, :target_seqlen + 1, :target_seqlen + 1]
			)

			if (allpos < 0).sum() > 0 or (allpos >= candidates_input.shape[1]).sum() > 0:
				print("???0")
				print((allpos < 0).sum())
				print((allpos >= candidates_input.shape[1]).sum())
			selected_candidates_output = torch.gather(candidates_output, 1,
					allpos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.max_skip, self.args.d_model))
			logits = torch.einsum("bld, bltd->blt", self.wq(hidden), self.wk(selected_candidates_output)) / self.sqrt_d_model
			logits += self.linear(hidden)
			# target = allpos[:, 1:] - allpos[:, :-1]
			target = torch.cat([allpos[:, 1:], origin_length.unsqueeze(-1) + 1], dim=1) - allpos

			gen_err = (target >= 4).sum().float() / target.shape[0]
			target = torch.min(target, torch.ones_like(target) * 3)

			if (target >= 4).sum() > 0 or (target < 0).sum():
				print("???1")

			length_loss = torch.nn.CrossEntropyLoss(reduction="none")(logits.transpose(1, 2), target)

			return dec_input, target_length, length_loss, gen_err
		else:
			if not use_inference_z:
				return origin_input, origin_length

			# next_input = origin_input[:, 0]
			prev_states = None

			now_pos = torch.zeros(batch_size, dtype=torch.long, device=origin_input.device)
			unk_mask = torch.zeros(batch_size, 1, 1, dtype=torch.float32, device=origin_input.device)
			allpos_arr = [now_pos]

			with torch.no_grad():
				for k in range(self.args.max_length):
					if (now_pos >= candidates_input.shape[1]).sum() > 0 or (now_pos < 0).sum() > 0:
						print("???2")
					next_input = torch.gather(candidates_input, 1, now_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.args.d_model))
					next_input = next_input * (1 - unk_mask) + \
								unk_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1) * unk_mask

					hidden, prev_states = self.decoder.incremental_forward(
						next_input + self.pos_embed(pos_idx[:, k]).unsqueeze(1), memory,
						src_mask, tgt_mask[:, :, k:k+1, :k+1],
						prev_states
					)

					# candidates_idx = torch.stack([now_pos + i for i in range(self.max_skip - 1)], dim=1).\
					# 	unsqueeze(-1).expand(-1, -1, candidates_input.shape[-1])
					# if (candidates_idx >= candidates_input.shape[1]).sum() > 0:
					# 	print("???3")
					candidates = torch.gather(candidates_output, 1,
							now_pos.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.max_skip, candidates_output.shape[-1]))
					logits = torch.einsum("bd, bjd->bj", self.wq(hidden[:, 0]), self.wk(candidates[:, 0])) / self.sqrt_d_model
					logits += self.linear(hidden[:, 0])
					if k == 0:
						logits[:, 0] = -1e9
						logits = logits - src_mask_float[:, :4, 0] * 1e9

					delta = None
					if self.args.z_decode_mode == "max":
						delta = logits.max(dim=-1)[1]
					elif self.args.z_decode_mode == "sample":
						delta = torch.multinomial(logits.softmax(dim=-1), 1)[:, 0]
					else:
						raise NotImplementedError("no such z_decode_mode")

					now_pos = torch.min(now_pos + delta, torch.ones(1, device=logits.device, dtype=torch.long) * (origin_seqlen + 1))
					unk_mask = (delta == 0).unsqueeze(-1).unsqueeze(-1).float()
					allpos_arr.append(now_pos)

					if k % 5 == 0 and (now_pos >= (origin_length + 1)).sum().tolist() == batch_size:
						break

			allpos = torch.stack(allpos_arr, dim=1)
			if (allpos >= candidates_input.shape[1]).sum() > 0:
				print("???4")

			unk_mask = ((allpos - torch.cat([torch.ones_like(allpos[:, :1]) * -1, allpos[:, :-1]], dim=1)) == 0).float().unsqueeze(-1)
			dec_input = torch.gather(candidates_input, 1, allpos.unsqueeze(-1).expand(-1, -1, candidates_input.shape[-1])) * (1 - unk_mask) +\
						unk_embed.unsqueeze(0).unsqueeze(0).expand(allpos.shape[0], allpos.shape[1], -1) * unk_mask
			#new_input = torch.gather(candidates_input, 1, allpos.unsqueeze(-1).expand(-1, -1, candidates_input.shape[-1]))
			target_length = (allpos < (origin_length + 1).unsqueeze(-1)).sum(dim=1) - 1

			return dec_input[:, :target_length.max() + 1], target_length, allpos

