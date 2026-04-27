import numpy as np
import cv2
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


def main():
    
    device = torch.device('cuda')
    
    net = unet.UNet(output_channel_count=3, min_channel_shape=(20, 25)).to(device)
    
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(net.input_shape),
            torchvision.transforms.ToTensor()
        ]
    )
    
    seg_map_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(net.output_shape),
            torchvision.transforms.ToTensor()
        ]
    )
    
    trainval_dataset = torchvision.datasets.OxfordIIITPet(
        root='./data',
        split='trainval',
        target_types='segmentation',
        transform=image_transform,
        target_transform=seg_map_transform,
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
