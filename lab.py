from dataclasses import dataclass
from typing      import Any, Optional

import pathlib

import torch
import torchvision
import monai
import matplotlib.pyplot as plt

import net_training as nt
import unet


def to_dataloader_for_tuples(applier, net, dataloader):
    
    dataloader_len = len(dataloader)
    dataset_len    = len(dataloader.dataset)
    
    losses  = torch.empty(dataloader_len, dtype=torch.float32)
    weights = torch.empty(dataloader_len, dtype=torch.float32)
    
    for index, batch in enumerate(dataloader):
        losses [index] = applier.to_batch(net, batch)
        weights[index] = batch[0].shape[0] / dataset_len
    
    return losses @ weights


class CrossEntropyLossApplier(nt.AbstractLossApplier):
    
    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss()
    
    def to_batch(self, net, batch):
        inputs, targets = batch
        return self.loss(net.forward(inputs), targets)
    
    def to_dataloader(self, net, dataloader):
        return to_dataloader_for_tuples(self, net, dataloader)


class MONAILossApplier(nt.AbstractLossApplier):
    
    def __init__(self, monai_loss):
        self.loss = monai_loss
    
    def to_batch(self, net, batch):
        
        inputs, targets = batch
        
        targets_new_shape = (targets.shape[0], 1, targets.shape[1], targets.shape[2])
        targets_one_hot = monai.networks.one_hot(targets.reshape(targets_new_shape), num_classes=3)
        
        return self.loss(net.forward(inputs), targets_one_hot)
    
    def to_dataloader(self, net, dataloader):
        return to_dataloader_for_tuples(self, net, dataloader)


class MONAIMetricApplier(nt.AbstractLossApplier):
    
    def __init__(self, monai_metric):
        self.metric = monai_metric
    
    def to_batch(self, net, batch):
        
        inputs, targets = batch
        
        outputs_one_hot = monai.networks.one_hot(net.forward(inputs).argmax(dim=1, keepdim=True), num_classes=3)
        
        targets_new_shape = (targets.shape[0], 1, targets.shape[1], targets.shape[2])
        targets_one_hot = monai.networks.one_hot(targets.reshape(targets_new_shape), num_classes=3)
        
        return self.metric(outputs_one_hot, targets_one_hot).mean()
    
    def to_dataloader(self, net, dataloader):
        return to_dataloader_for_tuples(self, net, dataloader)


class IoULossApplier(MONAILossApplier):
    
    def __init__(self):
        super().__init__(monai_loss=monai.losses.DiceLoss(softmax=True, jaccard=True))


class DiceLossApplier(MONAILossApplier):
    
    def __init__(self):
        super().__init__(monai_loss=monai.losses.DiceLoss(softmax=True))


class HausdorffDistanceLossApplier(MONAILossApplier):
    
    def __init__(self):
        super().__init__(monai_loss=monai.losses.HausdorffDTLoss(softmax=True))


class IoUMetricApplier(MONAIMetricApplier):
    
    def __init__(self):
        super().__init__(monai_metric=monai.metrics.MeanIoU())


class DiceMetricApplier(MONAIMetricApplier):
    
    def __init__(self):
        super().__init__(monai_metric=monai.metrics.DiceMetric())


class HausdorffDistanceMetricApplier(MONAIMetricApplier):
    
    def __init__(self):
        super().__init__(monai_metric=monai.metrics.HausdorffDistanceMetric())


class AllMetricsApplier(nt.AbstractLossApplier):
    
    def __init__(self):
        
        self.miou_loss = monai.losses.DiceLoss(softmax=True, jaccard=True)
        self.dice_loss = monai.losses.DiceLoss(softmax=True)
        self.haus_loss = monai.losses.HausdorffDTLoss(softmax=True)
        self.miou_metric = monai.metrics.MeanIoU()
        self.dice_metric = monai.metrics.DiceMetric()
        self.haus_metric = monai.metrics.HausdorffDistanceMetric()
    
    def to_batch(self, net, batch):
        
        inputs, targets = batch
        
        outputs = net.forward(inputs)
        outputs_one_hot = monai.networks.one_hot(outputs.argmax(dim=1, keepdim=True), num_classes=3)
        
        targets_new_shape = (targets.shape[0], 1, targets.shape[1], targets.shape[2])
        targets_one_hot = monai.networks.one_hot(targets.reshape(targets_new_shape), num_classes=3)
        targets_one_hot_cpu = targets_one_hot.cpu()
        
        return {
            'SoftIoU'          : 1 - self.miou_loss(output, targets_one_hot),
            'SoftDice'         : 1 - self.dice_loss(output, targets_one_hot),
            'HausdorffDTLoss'  : self.haus_loss(output.cpu(), targets_one_hot_cpu),
            'IoU'              : self.miou_metric(outputs_one_hot, targets_one_hot).mean(),
            'Dice'             : self.dice_metric(outputs_one_hot, targets_one_hot).mean(),
            'HausdorffDistance': self.haus_metric(outputs_one_hot.cpu(), targets_one_hot_cpu).mean()
        }
    
    def to_dataloader(self, net, dataloader):
        
        dataloader_len = len(dataloader)
        dataset_len    = len(dataloader.dataset)
        
        soft_miou = torch.empty(dataloader_len, dtype=torch.float32)
        soft_dice = torch.empty(dataloader_len, dtype=torch.float32)
        haus_loss = torch.empty(dataloader_len, dtype=torch.float32)
        miou      = torch.empty(dataloader_len, dtype=torch.float32)
        dice      = torch.empty(dataloader_len, dtype=torch.float32)
        haus_dist = torch.empty(dataloader_len, dtype=torch.float32)
        
        weights = torch.empty(dataloader_len, dtype=torch.float32)
        
        for i, batch in enumerate(dataloader):
            
            batch_result = self.to_batch(net, batch)
            
            soft_miou[i] = batch_result['SoftIoU']
            soft_dice[i] = batch_result['SoftDice']
            haus_loss[i] = batch_result['HausdorffDTLoss']
            miou     [i] = batch_result['IoU']
            dice     [i] = batch_result['Dice']
            haus_dist[i] = batch_result['HausdorffDistance']
            
            weights[i] = batch[0].shape[0] / dataset_len
        
        return {
            'SoftIoU'          : soft_miou @ weights,
            'SoftDice'         : soft_dice @ weights,
            'HausdorffDTLoss'  : haus_loss @ weights,
            'IoU'              : miou      @ weights,
            'Dice'             : dice      @ weights,
            'HausdorffDistance': haus_dist @ weights
        }


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


def main(output_dir='./'):
    
    if not isinstance(output_dir, pathlib.Path):
        output_dir = pathlib.Path(output_dir)
    
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
        batch_size=5,
        shuffle=True,
        #pin_memory=True
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=5,
        #pin_memory=True
    )
    
    def dict_format(d):
        if isinstance(d, float):
            return f'{d:.3f}'
        elif isinstance(d, dict):
            return '{ ' + ', '.join([f'{k}: {dict_format(v)}' for k, v in d.items()]) + ' }'
        else:
            return str(d)
    
    epoch_count = 10
    loss_name = 'CrossEntropyLoss'
    
    train_test_result = nt.NetTrainer(
        loss_applier=CrossEntropyLossApplier(),
        optimizer_factory=AdamOptimizerFactory(),
        iteration_logger=nt.IterationLogger(
            message_sender=lambda count, time, info: print(f'[{time:%Y-%m-%d %H:%M:%S}] iteration {count}: {dict_format(info)}'),
            duration=1
        ),
        epoch_logger=nt.IterationLogger(
            message_sender=lambda count, time, info: print(f'[{time:%Y-%m-%d %H:%M:%S}] epoch {count}: {dict_format(info)}'),
            duration=1
        )
    ).train_test(
        net=net,
        train_dataloader=trainval_dataloader,
        n_epochs=epoch_count,
        test_dataloader=test_dataloader,
        metric_applier=AllMetricsApplier(),
        #gc_collect=True,
        #cuda_cache_clear=True
    )
    
    epoch_range = list(range(epoch_count))
    
    train_loss_history = train_test_result['train_loss_history']
    test_loss_history  = train_test_result[ 'test_loss_history']
    
    train_iou_history = [ e['IoU'] for e in train_test_result['train_metric_history'] ]
    test_iou_history  = [ e['IoU'] for e in train_test_result[ 'test_metric_history'] ]
    train_soft_iou_history = [ e['SoftIoU'] for e in train_test_result['train_metric_history'] ]
    test_soft_iou_history  = [ e['SoftIoU'] for e in train_test_result[ 'test_metric_history'] ]
    
    train_dice_history = [ e['Dice'] for e in train_test_result['train_metric_history'] ]
    test_dice_history  = [ e['Dice'] for e in train_test_result[ 'test_metric_history'] ]
    train_soft_dice_history = [ e['SoftDice'] for e in train_test_result['train_metric_history'] ]
    test_soft_dice_history  = [ e['SoftDice'] for e in train_test_result[ 'test_metric_history'] ]
    
    train_haus_loss_history = [ e['HausdorffDTLoss'] for e in train_test_result['train_metric_history'] ]
    test_haus_loss_history  = [ e['HausdorffDTLoss'] for e in train_test_result[ 'test_metric_history'] ]
    
    train_haus_dist_history = [ e['HausdorffDistance'] for e in train_test_result['train_metric_history'] ]
    test_haus_dist_history  = [ e['HausdorffDistance'] for e in train_test_result[ 'test_metric_history'] ]
    
    plt.plot(epoch_range, train_loss_history, label='train loss')
    plt.plot(epoch_range,  test_loss_history, label= 'test loss')
    plt.legend()
    plt.savefig(output_dir/f'{loss_name}.png')
    
    plt.plot(epoch_range,      train_iou_history, label='train IoU')
    plt.plot(epoch_range,       test_iou_history, label= 'test IoU')
    plt.plot(epoch_range, train_soft_iou_history, label='train soft IoU')
    plt.plot(epoch_range,  test_soft_iou_history, label= 'test soft IoU')
    plt.legend()
    plt.savefig(output_dir/'metrics'/'IoU.png')
    
    plt.plot(epoch_range,      train_dice_history, label='train Dice')
    plt.plot(epoch_range,       test_dice_history, label= 'test Dice')
    plt.plot(epoch_range, train_soft_dice_history, label='train soft Dice')
    plt.plot(epoch_range,  test_soft_dice_history, label= 'test soft Dice')
    plt.legend()
    plt.savefig(output_dir/'metrics'/'Dice.png')
    
    plt.plot(epoch_range, train_haus_loss_history, label='train Hausdorff DT loss')
    plt.plot(epoch_range,  test_haus_loss_history, label= 'test Hausdorff DT loss')
    plt.legend()
    plt.savefig(output_dir/'metrics'/'HausdorffDTLoss.png')
    
    plt.plot(epoch_range, train_haus_dist_history, label='train Hausdorff distance')
    plt.plot(epoch_range,  test_haus_dist_history, label= 'test Hausdorff distance')
    plt.legend()
    plt.savefig(output_dir/'metrics'/'HausdorffDistance.png')


if __name__ == '__main__':
    main()
