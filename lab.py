from dataclasses import dataclass
from typing      import Optional

import numpy as np
import torch
import torchvision

import net_training as nt
import unet


class CrossEntropyLossApplier(nt.AbstractLossApplier):
    
    def __init__(self):
        pass
    
    def __call__(self, net, batch):
        inputs, targets = batch
        return torch.nn.functional.cross_entropy(net.forward(inputs), targets)


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

    result_dtype  : torch.dtype
    result_shape  : Optional[tuple[int, int]] = None
    result_device : Optional[torch.device] = None

    def __call__(self, pil_image):
        t = torchvision.transforms.functional.pil_to_tensor(pil_image)
        if self.result_shape is not None:
            t = torchvision.transforms.functional.resize(t, size=self.result_shape)
        if self.result_device is not None:
            t = t.to(self.result_device)
        t = t.to(self.result_dtype)
        return t


def main():
    
    device = torch.device('cuda')
    
    net = unet.UNet(output_channel_count=3, min_channel_shape=(20, 25)).to(device)
    
    trainval_dataset = torchvision.datasets.OxfordIIITPet(
        root='./',
        split='trainval',
        target_types='segmentation',
        transform=PILImageToTensorTransform(
            result_dtype=torch.float32,
            result_shape=net.input_shape,
            result_device=device
        ),
        target_transform=PILImageToTensorTransform(
            result_dtype=torch.int64,
            result_shape=net.output_shape,
            result_device=device
        ),
        download=True
    )
    
    trainval_dataloader = torch.utils.data.DataLoader(batch_size=64, shuffle=True, pin_memory=True)
    
    training_result = nt.NetTrainer(
        loss_applier=CrossEntropyLossApplier(),
        optimizer_factory=AdamOptimizerFactory(),
        epoch_logger=nt.IterationLogger(
            message_sender=lambda x: print(f'epoch {x} completed'),
            duration=1
        )
    ).train(
        net=net,
        dataloader=trainval_dataloader,
        n_epochs=10
    )


if __name__ == '__main__':
    main()
