from abc import ABC, abstractmethod

import torch


class AbstractLossApplier(ABC):

	@abstractmethod
	def __call__(self, net, batch):
		pass
