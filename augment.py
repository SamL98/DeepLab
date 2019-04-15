from enum import Enum
from skimage.transform import resize
import numpy as np
import util

class BrightnessDirection(Enum):
	Stay = 'none'
	Up = 'up'
	Down = 'down'

GAMMA_DOWN = 0.7
GAMMA_UP = 1.4
RES_DOWN = 0.8
	
def augment(rgb, flip_lr, scale_down, brightness_direction):
	if flip_lr:
		rgb = np.fliplr(rgb)
		
	if scale_down:
		h, w, _ = rgb.shape
		new_h = int(h*RES_DOWN)
		new_w = int(w*RES_DOWN)
		rgb = resize(rgb/255., (new_h, new_w))
		rgb = (resize(rgb, (h, w))*255).astype(np.uint8)
			
	if brightness_direction != BrightnessDirection.Stay:
		gamma = GAMMA_DOWN
		if brightness_direction == BrightnessDirection.Up:
			gamma = GAMMA_UP
		
		rgb = ((rgb/255.)**(1/gamma) * 255).astype(np.uint8)
		
	return rgb

def generate_augmentation_volume(rgb):
	num_aug = 12
	rgb_aug = np.zeros((num_aug, *rgb.shape), dtype=np.uint8)

	aug_no = -1
	flip_idxs = []

	for lr in [False, True]:
		for scale in [False, True]:
			for bd in [BrightnessDirection.Stay, BrightnessDirection.Up, BrightnessDirection.Down]:
				aug_no += 1

				if (not lr) and (not scale) and (bd == BrightnessDirection.Stay):
					rgb_aug[aug_no,...] = rgb.copy()
					continue

				if lr:
					flip_idxs.append(aug_no)

				new_rgb = augment(rgb.copy(), lr, scale, bd)
				rgb_aug[aug_no,...] = new_rgb

	return rgb_aug, flip_idxs

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
