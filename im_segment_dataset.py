import cv2
import torch


class ImageSegmentDataset(torch.utils.data.Dataset):

	def __init__(self, image_paths, seg_map_paths):
		if len(image_paths) != len(seg_map_paths):
			raise ValueError(
				'len(image_paths) != len(seg_map_paths)\n'
				'where\n'
				f'\t{len(image_paths)=}\n'
				f'\t{len(seg_map_paths)=}'
			)
		self.image_paths = image_paths
		self.seg_map_paths = seg_map_paths

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		image   = cv2.resize(cv2.imread(self.  image_paths[index]), dsize=(600, 400))
		seg_map = cv2.resize(cv2.imread(self.seg_map_paths[index], cv2.IMREAD_GRAYSCALE), dsize=(600, 400)) - 1
		image_tensor   = torch.tensor(image.transpose((2, 0, 1)).astype(np.float32))
		seg_map_tensor = torch.tensor(seg_map.astype(np.int64))
		return image_tensor, seg_map_tensor
