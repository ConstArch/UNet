from dataclasses import dataclass

import numpy as np
import torch

import loss_applying as lapp
import optim_factory as optf
import iter_logging as il


@dataclass
class NetTrainer:
    
    loss_applier      : lapp.AbstractLossApplier
    optimizer_factory : optf.AbstractOptimizerFactory
    iteration_logger  : il.IterationLogger
    epoch_logger      : il.IterationLogger
    
    def train(self, net, dataloader, n_epochs):
        
        net.train(True)
        
        optimizer = self.optimizer_factory.with_parameters(net.parameters())
        loss_history = []
        
        # begin outer for
        for _ in range(n_epochs):
            
            # begin inner for
            for batch in dataloader:
                
                loss = self.loss_applier(net, batch)
                
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
    
    def train_valid(self, net, dataloader, n_epochs, dataset_valid, metric):
        
        net.train(True)
        
        optimizer = self.optimizer_factory.with_parameters(net.parameters())
        loss_history_train = []
        loss_history_valid = []
        metric_history_train = []
        metric_history_valid = []
        
        # begin outer for
        for _ in range(n_epochs):
            
            # begin inner for
            for batch in dataloader:
                
                loss = self.loss_applier(net, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if self.iteration_logger is not None:
                    self.iteration_logger.tick()
                
            # end inner for
            
            loss_valid = self.loss_applier(net, dataset_valid)
            
            loss_history_train.append(loss.item())
            loss_history_valid.append(loss_valid.item())
            metric_history_train.append(metric(dataloader.dataset.tensor_Y, net.forward(dataloader.dataset.tensor_X)))
            metric_history_valid.append(metric(dataset_valid.tensor_Y, net.forward(dataset_valid.tensor_X)))
            
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
            'loss_history_train': loss_history_train,
            'loss_history_valid': loss_history_valid,
            'metric_history_train': metric_history_train,
            'metric_history_valid': metric_history_valid
        }
        
    # end NetTrainer.train_valid
    
# end NetTrainer
