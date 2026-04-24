from abc import ABC, abstractmethod

import torch


class AbstractOptimizerFactory(ABC):

    @abstractmethod
    def with_parameters(self, parameters):
        pass


class SGDOptimizerFactory(AbstractOptimizerFactory):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def with_parameters(self, parameters):
        return torch.optim.SGD(**self.kwargs, params=parameters)


class AdamOptimizerFactory(AbstractOptimizerFactory):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def with_parameters(self, parameters):
        return torch.optim.Adam(**self.kwargs, params=parameters)
