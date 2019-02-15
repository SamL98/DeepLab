from hdf5storage import loadmat
from os.path import join
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

from util import read_slices
from perform_calibrated_inference import calibrate_logits

slices = read_slices('slices.pkl')

import sys
if len(sys.argv) > 1:
	idx = int(sys.argv[1])
else:
	m = 350#1449-350
	idx = np.random.choice(m, 1)[0]
	print(idx)

ds_path = 'D:/datasets/processed/voc2012'
class_labels = loadmat(join(ds_path, 'dataset_info.mat'))['class_labels']

imset = 'test'
rgb_path = join(ds_path, 'rgb', imset, imset+'_%06d_rgb.jpg')
gt_path = join(ds_path, 'truth', imset, imset+'_%06d_pixeltruth.mat')
dl_pred_path = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_prediction.mat')
#calib_pred_path = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_calib_pred.mat')


img = imread(rgb_path % idx)
gt = loadmat(gt_path % idx)['truth_img']
gt[gt==255] = 0
dl_pred = loadmat(dl_pred_path % idx)['pred_img']+1
dl_pred[(gt==0)] = 0
#calib_pred = loadmat(calib_pred_path % idx)['pred_img']

conf_thresh = 0.75
if len(sys.argv) > 2:
	conf_thresh = float(sys.argv[2])
	
downsample_factor = 1
	
calib_pred, calib_conf = calibrate_logits(idx, imset, slices, len(slices[0][0].acc_hist), conf_thresh=conf_thresh, ret_conf=True, ds_factor=downsample_factor)

print('GT Labels:')
for lab in np.unique(gt):
	print('%d: %s' % (lab, class_labels[lab]))
print('***')

print('DL Labels:')
for lab in np.unique(dl_pred):
	print('%d: %s' % (lab, class_labels[lab]))
print('***')
	
print('CALIB Labels:')
for lab in np.unique(calib_pred):
	if lab == 0: 
		print('0: background')
		continue
		
	base = 0
	for slice in slices:
		if lab-base <= len(slice):
			print('%d: %s' % (lab, slice[lab-base-1].name))
			break
		base += len(slice)
print('***')


fig, ax = plt.subplots(2, 2)

ax[0,0].imshow(img)
ax[0,0].set_title('Input')
ax[0,0].axis('off')

ax[0,1].imshow(gt, 'jet', vmin=0, vmax=20)
ax[0,1].set_title('Ground Truth')
ax[0,1].axis('off')

ax[1,0].imshow(dl_pred, 'jet', vmin=0, vmax=20)
#ax[1,0].imshow(calib_conf, 'gray', vmin=0, vmax=1)
ax[1,0].set_title('DeepLab Pred')
ax[1,0].axis('off')

ax[1,1].imshow(calib_pred, 'jet', vmin=0, vmax=20)
ax[1,1].set_title('Calibrated Pred')
ax[1,1].axis('off')

fig.savefig('images/vis_calib_%s_%d.png' % (imset, idx), bbox_inches='tight')
plt.show()