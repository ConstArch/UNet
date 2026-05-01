from abc         import ABC, abstractmethod
from dataclasses import dataclass
from typing      import Optional

#import numpy as np
import torch


class AbstractLossApplier(ABC):
    
    @abstractmethod
    def to_batch(self, net, batch):
        pass
    
    @abstractmethod
    def to_dataloader(self, net, dataloader):
        pass


class AbstractOptimizerFactory(ABC):
    
    @abstractmethod
    def with_parameters(self, parameters):
        pass


class IterationLogger:
    
    def __init__(self, message_sender, duration):
        self.message_sender = message_sender
        self.duration = duration
        self.count = 0
    
    def tick(self):
        self.count += 1
        if self.count % self.duration == 0:
            self.message_sender(self.count)
    
    def reset(self):
        self.count = 0


def load_all(dataset, collate_fn=torch.utils.data.default_collate):
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
    
    return list(dataloader)[0]


@dataclass
class NetTrainer:
    
    loss_applier      : AbstractLossApplier
    optimizer_factory : AbstractOptimizerFactory
    iteration_logger  : Optional[IterationLogger] = None
    epoch_logger      : Optional[IterationLogger] = None
    
    def train(self, net, dataloader, n_epochs):
        
        net.train(True)
        
        optimizer = self.optimizer_factory.with_parameters(net.parameters())
        loss_history = []
        
        # begin outer for
        for _ in range(n_epochs):
            
            # begin inner for
            for batch in dataloader:
                
                loss = self.loss_applier.to_batch(net, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if self.iteration_logger is not None:
                    self.iteration_logger.tick()
                
            # end inner for
            
            loss_history.append(loss.item())
            
            if self.iteration_logger is not None:
                self.iteration_logger.reset()
            
            if self.epoch_logger is not None:
                self.epoch_logger.tick()
            
        # end outer for
        
        if self.epoch_logger is not None:
            self.epoch_logger.reset()
        
        net.train(False)
        
        return { 'net': net, 'loss_history': loss_history }
        
    # end NetTrainer.train
    
    def train_test(self, net, train_dataloader, n_epochs, test_dataloader, metric_applier):
        
        net.train(True)
        
        optimizer = self.optimizer_factory.with_parameters(net.parameters())
        train_loss_history   = []
        test_loss_history    = []
        train_metric_history = []
        test_metric_history  = []
        
        # begin outer for
        for _ in range(n_epochs):
            
            # begin inner for
            for batch in train_dataloader:
                
                loss = self.loss_applier.to_batch(net, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if self.iteration_logger is not None:
                    self.iteration_logger.tick()
                
            # end inner for
            
            train_loss   = self.loss_applier.to_dataloader(net, train_dataloader)
            test_loss    = self.loss_applier.to_dataloader(net,  test_dataloader)
            train_metric =    metric_applier.to_dataloader(net, train_dataloader)
            test_metric  =    metric_applier.to_dataloader(net,  test_dataloader)
            
            train_loss_history  .append(train_loss.item())
            test_loss_history   .append( test_loss.item())
            train_metric_history.append(train_metric)
            test_metric_history .append( test_metric)
            
            if self.iteration_logger is not None:
                self.iteration_logger.reset()
            
            if self.epoch_logger is not None:
                self.epoch_logger.tick()
            
        # end outer for
        
        if self.epoch_logger is not None:
            self.epoch_logger.reset()
        
        net.train(False)
        
        return {
            'net': net,
            'train_loss_history'  : train_loss_history,
            'test_loss_history'   :  test_loss_history,
            'train_metric_history': train_metric_history,
            'test_metric_history' :  test_metric_history
        }
        
    # end NetTrainer.train_valid
    
# end NetTrainer
