from hdf5storage import loadmat
from os.path import join
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

from util import *
from perform_calibrated_inference import calibrate_logits

slices = read_slices('slices.pkl')
imset = 'test'

import sys
if len(sys.argv) > 1:
	idx = int(sys.argv[1])
else:
	m = num_img_for(imset)
	idx = np.random.choice(m, 1)[0]
	print(idx)

img = load_rgb(imset, idx)
gt = load_gt(imset, idx, reshape=False)
gt[gt==255] = 0

dl_pred = load_dl_pred(imset, idx)
dl_pred[(gt==0)] = 0

conf_thresh = 0.75
if len(sys.argv) > 2:
	conf_thresh = float(sys.argv[2])
	

calib_pred = load_calib_pred(imset, idx, conf_thresh)
if calib_pred is None:
	calib_pred = calibrate_logits(idx, imset, slices, len(slices[0][0].acc_hist), True, conf_thresh=conf_thresh, sm_by_slice=False)

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

max_label = calib_pred.max()

fig, ax = plt.subplots(2, 2)

ax[0,0].imshow(img)
ax[0,0].set_title('Input')
ax[0,0].axis('off')

ax[0,1].imshow(gt, 'jet', vmin=0, vmax=max_label)
ax[0,1].set_title('Ground Truth')
ax[0,1].axis('off')

ax[1,0].imshow(dl_pred, 'jet', vmin=0, vmax=max_label)
ax[1,0].set_title('DeepLab Pred')
ax[1,0].axis('off')

ax[1,1].imshow(calib_pred, 'jet', vmin=0, vmax=max_label)
ax[1,1].set_title('Calibrated Pred')
ax[1,1].axis('off')

fig.savefig('images/vis_calib_%s_%d.png' % (imset, idx), bbox_inches='tight')
plt.show()