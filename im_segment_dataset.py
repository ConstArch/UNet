import numpy as np
import cv2
import torch


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
		
	#end def __init__

	def __len__(self):
		return len(self.image_paths)
	#end def __len__

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
		
	#end def __getitem__
	
#end class ImageSegmentDataset
