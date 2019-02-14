from hdf5storage import loadmat
from os.path import join
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

import sys
if len(sys.argv) > 1:
	idx = int(sys.argv[1])
else:
	m = 1449-350
	idx = np.random.choice(m, 1)[0]

ds_path = 'D:/datasets/processed/voc2012'
imset = 'test'
rgb_path = join(ds_path, 'rgb', imset, imset+'_%06d_rgb.jpg')
gt_path = join(ds_path, 'truth', imset, imset+'_%06d_pixeltruth.mat')
dl_pred_path = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_prediction.mat')
calib_pred_path = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_calib_pred.mat')




img = imread(rgb_path % idx)
gt = loadmat(gt_path % idx)['truth_img']
dl_pred = loadmat(dl_pred_path % idx)['pred_img']
calib_pred = loadmat(calib_pred_path % idx)['pred_img']


fig, ax = plt.subplots(1, 4)

ax[0].imshow(img)
ax[0].set_title('Input')
ax[0].axis('off')

ax[1].imshow(gt, 'jet', vmin=0, vmax=255)
ax[1].set_title('Ground Truth')
ax[1].axis('off')

ax[2].imshow(dl_pred, 'jet', vmin=0, vmax=255)
ax[2].set_title('DeepLab Pred')
ax[2].axis('off')

ax[3].imshow(calib_pred, 'jet', vmin=0, vmax=255)
ax[3].set_title('Calibrated Pred')
ax[3].axis('off')

fig.savefig('images/vis_calib_%s_%d.png' % (imset, idx), bbox_inches='tight')
plt.show()