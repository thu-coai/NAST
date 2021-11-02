import numpy as np

from cotk.metric import MetricBase

class NameChanger(MetricBase):
	def __init__(self, target, prefix):
		self.target = target
		self.prefix = prefix

	def forward(self, data):
		self.target.forward(data)

	def close(self):
		return {self.prefix + key: value for key, value in self.target.close().items()}