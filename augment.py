from enum import Enum
from collections import namedtuple
import numpy as np
import util

class BrightnessDirection(Enum):
	Stay = 'none'
	Up = 'up'
	Down = 'down'
	
Bbox = namedtuple('bbox', 'y x h w')

def _random_crop_bbox(fg_bbox, h, w):
	x = 0
	if fg_bbox.x > 0:
		x = np.random.randint(fg_bbox.x)
		
	max_x = w
	if fg_bbox.x+fg_bbox.w < w:
		max_x = np.random.randint(fg_bbox.x+fg_bbox.w, high=w)
		
	y = 0
	if fg_bbox.y > 0:
		y = np.random.randint(fg_bbox.y)
		
	max_y = h
	if fg_bbox.y+fg_bbox.h < h:
		max_y = np.random.randint(fg_bbox.y+fg_bbox.h, high=h)
	return Bbox(y, x, max_y-y, max_x-x)

def augment(rgb, gt, flip_lr, crop, brightness_direction, min_gamma=0.6, max_gamma=1.6):
	if flip_lr:
		rgb = np.fliplr(rgb)
		gt = np.fliplr(gt)
		
	if crop:
		fg_mask = util.fg_mask_for(gt)
		if fg_mask.sum() == 0:
			return None, None
		
		
		try:
			where_fg_row, where_fg_col = np.where(fg_mask)
		except ValueError:
			util.stdout_writeln(np.where(fg_mask).__repr__())
			exit()
			
		fg_bbox_x = where_fg_col.min()
		fg_bbox_y = where_fg_row.min()
		fg_bbox = Bbox(fg_bbox_y, fg_bbox_x, where_fg_row.max()-fg_bbox_y, where_fg_col.max()-fg_bbox_x)
		
		crop_bbox = _random_crop_bbox(fg_bbox, *gt.shape)
		rgb = rgb[crop_bbox.y:crop_bbox.y+crop_bbox.h, crop_bbox.x:crop_bbox.x+crop_bbox.w]
		gt = gt[crop_bbox.y:crop_bbox.y+crop_bbox.h, crop_bbox.x:crop_bbox.x+crop_bbox.w]
			
	if brightness_direction != BrightnessDirection.Stay:
		lo, hi = min_gamma, 0.8
		if brightness_direction == BrightnessDirection.Up:
			lo, hi = 1.25, max_gamma
		
		gamma = np.random.uniform(lo, hi)
		rgb = ((rgb/255.)**(1/gamma) * 255).astype(np.uint8)
		
	return rgb, gt
	

if __name__ == '__main__':
	idx = np.random.randint(1, high=util.num_img_for('val')+1)
	rgb = util.load_rgb('val', idx)
	gt = util.load_gt('val', idx)
	
	rgb_aug, gt_aug = augment(rgb, gt, False, True, BrightnessDirection.Down)
	
	import matplotlib.pyplot as plt
	cmap = util.voc_colormap()
	
	_, ax = plt.subplots(2, 2)
	ax[0,0].imshow(rgb)
	ax[0,1].imshow(cmap[gt])
	ax[1,0].imshow(rgb_aug)
	ax[1,1].imshow(cmap[gt_aug])
	plt.show()