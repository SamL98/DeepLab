import numpy as np
from hdf5storage import loadmat
import os
from os.path import join

import sys
imset = sys.argv[1].lower().capitalize()

ROOT_DIR = 'D:/Datasets/Processed/VOC2012'
PRED_DIR = join(ROOT_DIR, 'Deeplab_Prediction/' + imset)
GT_DIR = join(ROOT_DIR, 'Truth/' + imset)

PRED_FMT = imset.lower()+'_%06d_prediction.mat'
GT_FMT = imset.lower()+'_%06d_pixeltruth.mat'

ds_info = loadmat(join(ROOT_DIR, 'dataset_info.mat'))
num_class = ds_info['num_labels']-1
i_per_class = np.zeros(num_class, dtype=np.float64)
u_per_class = np.zeros(num_class, dtype=np.float64)
num_img = ds_info['num_'+imset.lower()]

for idx in range(1, num_img+1):
	pred = loadmat(join(PRED_DIR, PRED_FMT % idx))['pred_img']
	gt = loadmat(join(GT_DIR, GT_FMT % idx))['truth_img']
	
	for i in range(num_class):
		pred_mask = pred==i
		gt_mask = gt==i
		i_per_class[i] += np.sum(pred_mask & gt_mask)
		u_per_class[i] += np.sum(pred_mask | gt_mask)
		
		
ious_per_class = i_per_class / np.maximum(1e-8, u_per_class)
mIOU = np.sum(ious_per_class)/num_class

print('----------------')
print('mIOU: %f' % mIOU)
print('----------------')