from dataclasses import dataclass
from typing      import Any, Optional

import numpy as np
import torch
import torchvision
import torchmetrics
import matplotlib.pyplot as plt

import net_training as nt
import unet


class CrossEntropyLossApplier(nt.AbstractLossApplier):
    
    def __init__(self):
        pass
    
    def to_batch(self, net, batch):
        inputs, targets = batch
        return torch.nn.functional.cross_entropy(net.forward(inputs), targets)
    
    def to_dataloader(self, net, dataloader):
        
        N = len(dataloader)
        
        losses  = torch.empty(N, dtype=torch.float32)
        weights = torch.empty(N, dtype=torch.float32)
        
        for index, batch in enumerate(dataloader):
            losses [index] = self.to_batch(net, batch)
            weights[index] = batch[0].shape[0] / N
        
        return losses @ weights


class IoUMetricApplier:
    
    def __init__(self):
        pass
    
    def to_dataloader(self, net, dataloader):
        pass


class DiceMetricApplier:
    
    def __init__(self):
        pass
    
    def to_dataloader(self, net, dataloader):
        pass


class SGDOptimizerFactory(nt.AbstractOptimizerFactory):
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def with_parameters(self, parameters):
        return torch.optim.SGD(**self.kwargs, params=parameters)


class AdamOptimizerFactory(nt.AbstractOptimizerFactory):
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def with_parameters(self, parameters):
        return torch.optim.Adam(**self.kwargs, params=parameters)


@dataclass
class PILImageToTensorTransform:
    
    shape        : Optional[tuple[int, int]] = None
    scale        : Any = None
    offset       : Any = None
    dtype        : torch.dtype = None
    device       : Optional[torch.device] = None
    non_blocking : bool = False
    
    def __call__(self, pil_image):
        
        image_tensor = torchvision.transforms.functional.pil_to_tensor(pil_image)
        
        if self.shape is not None:
            image_tensor = torchvision.transforms.functional.resize(image_tensor, size=self.shape)
        
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor[0]
        
        if self.scale is not None:
            image_tensor *= self.scale
        
        if self.offset is not None:
            image_tensor += self.offset
        
        return image_tensor.to(dtype=self.dtype, device=self.device, non_blocking=self.non_blocking)


def main():
    
    device = torch.device('cuda')
    
    net = unet.UNet(output_channel_count=3, min_channel_shape=(20, 25)).to(device)
    
    image_transform = PILImageToTensorTransform(
        shape=net.input_shape,
        dtype=torch.float32,
        device=device,
        non_blocking=True
    )
    
    seg_map_transform = PILImageToTensorTransform(
        shape=net.output_shape,
        offset=-1,
        dtype=torch.int64,
        device=device,
        non_blocking=True
    )
    
    trainval_dataset = torchvision.datasets.OxfordIIITPet(
        root='./',
        split='trainval',
        target_types='segmentation',
        transform=image_transform,
        target_transform=seg_map_transform,
        download=True
    )
    
    test_dataset = torchvision.datasets.OxfordIIITPet(
        root='./',
        split='test',
        target_types='segmentation',
        transform=image_transform,
        target_transform=seg_map_transform,
        download=True
    )
    
    trainval_dataloader = torch.utils.data.DataLoader(
        trainval_dataset,
        batch_size=100,
        shuffle=True,
        pin_memory=True
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        pin_memory=True
    )
    
    epoch_count = 10
    
    train_test_result = nt.NetTrainer(
        loss_applier=CrossEntropyLossApplier(),
        optimizer_factory=AdamOptimizerFactory(),
        epoch_logger=nt.IterationLogger(
            message_sender=lambda x: print(f'epoch {x} completed'),
            duration=1
        )
    ).train_test(
        net=net,
        train_dataloader=trainval_dataloader,
        n_epochs=epoch_count,
        test_dataloader=test_dataloader,
        metric_applier=DiceMetricApplier()
    )
    
    epoch_range = list(range(epoch_count))
    
    plt.plot(epoch_range, train_test_result['train_loss_history'])
    plt.plot(epoch_range, train_test_result[ 'test_loss_history'])
    plt.show()
    
    plt.plot(epoch_range, train_test_result['train_metric_history'])
    plt.plot(epoch_range, train_test_result[ 'test_metric_history'])
    plt.show()


if __name__ == '__main__':
    main()
