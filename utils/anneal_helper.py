#coding:utf-8
import numpy as np

#TODO: make it to a hook, not hardcode in model_helper
class AnnealHelper:
	def __init__(self, instance, name, beginValue, startBatch, startValue, endValue, multi=1, linear_add=0):
		self.instance = instance
		self.name = name
		self.initial = startValue
		self.end = endValue
		self.startbatch = startBatch
		self.multi = multi
		self.linear_add = linear_add
		self.beginValue = beginValue

	def step(self):
		if self.instance.now_batch >= self.startbatch and self.instance.param.other_weights[self.name] == self.beginValue:
			self.instance.param.other_weights[self.name] = self.initial
			# fall down to make the weight != inital

		if self.instance.now_batch > self.startbatch:
			ori = self.instance.param.other_weights[self.name]
			self.instance.param.other_weights[self.name] = self.instance.param.other_weights[self.name] * self.multi + self.linear_add
			if self.instance.param.other_weights[self.name] > ori:
				if self.instance.param.other_weights[self.name] > self.end:
					self.instance.param.other_weights[self.name] = self.end
			else:
				if self.instance.param.other_weights[self.name] < self.end:
					self.instance.param.other_weights[self.name] = self.end

	def over(self):
		return np.isclose(self.instance.param.other_weights[self.name], self.end)

class AnnealParameter(tuple):
	'''Set parameter for AnnealHelper. BaseModel will automatically create AnnealHelper.
	'''
	@staticmethod
	def create_set(value):
		return AnnealParameter(("set", {"value": value}))

	@staticmethod
	def create_hold(value):
		return AnnealParameter(("hold", {"value": value}))

	@staticmethod
	def create_anneal(beginValue, startBatch, startValue, endValue, multi=1, linear_add=0):
		return AnnealParameter(("anneal", {"beginValue": beginValue, "startBatch": startBatch, "startValue": startValue, "endValue":endValue, "multi":multi, "linear_add":linear_add}))

	@staticmethod
	def create_set_and_anneal(startValue, endValue, multi):
		return AnnealParameter(("set&anneal", {"startValue": startValue, "endValue":endValue, "multi":multi}))
