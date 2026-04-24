import numpy as np
import cv2
import torch


import net_training as nt
import unet


class ImageSegmentDataset(torch.utils.data.Dataset):
    
    def __init__(self, image_paths, seg_map_paths, image_height, image_width):
        
        if len(image_paths) != len(seg_map_paths):
            raise ValueError(
                'len(image_paths) != len(seg_map_paths)\n'
                'where\n'
                f'\t{len(image_paths)=}\n'
                f'\t{len(seg_map_paths)=}'
            )
        
        self.image_paths   = image_paths
        self.seg_map_paths = seg_map_paths
        self.image_height  = image_height
        self.image_width   = image_width
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        
        image = cv2.resize(
            cv2.imread(self.image_paths[index]),
            dsize = (self.image_width, self.image_height)
        )
        seg_map = cv2.resize(
            cv2.imread(self.seg_map_paths[index], cv2.IMREAD_GRAYSCALE),
            dsize = (self.image_width, self.image_height)
        )
        
        image = image.transpose((2, 0, 1)).astype(np.float32)
        seg_map = (seg_map - 1).astype(np.int64)
        
        return torch.tensor(image), torch.tensor(seg_map)


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
    
    with open('Oxford pets/Segmentation/annotations/trainval.txt') as fin:
        train_names = [line.split()[0] for s in fin.readlines()]
    
    with open('Oxford pets/Segmentation/annotations/test.txt') as fin:
        test_names = [line.split()[0] for s in fin.readlines()]


if __name__ == '__main__':
    main()
