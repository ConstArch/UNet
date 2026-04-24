from abc import ABC, abstractmethod

import torch


class AbstractLossApplier(ABC):

	@abstractmethod
	def __call__(self, net, batch):
		pass


class CrossEntropyLossApplier(AbstractLossApplier):

	def __init__(self):
		pass

	def __call__(self, net, batch):
		inputs, targets = batch
		return torch.nn.functional.cross_entropy(net.forward(inputs), targets)
