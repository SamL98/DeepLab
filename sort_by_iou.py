import numpy as np
from hdf5storage import loadmat
from PIL import Image
from os.path import join

import sys
imset = sys.argv[1].lower()

HOME_DIR = 'D:/Datasets/Processed/VOC2012/'
GT_DIR = join(HOME_DIR, 'Truth', imset)
PRED_DIR = join(HOME_DIR, 'Deeplab_Prediction', imset)

def iou(pred, gt, nc):
	iou_per_class = []
	
	pred[gt==255] = 255
	valid_labels = np.unique(gt)
	valid_labels = np.concatenate((valid_labels, np.unique(pred)), axis=0)
	valid_labels = set((valid_labels[valid_labels!=255]).astype(np.uint8))
	
	for i in valid_labels:
		gt_mask = (gt==i)#.astype(np.uint8)
		pred_mask = (pred==i)#.astype(np.uint8)
		#iou_per_class.append((gt_mask & pred_mask).sum() / max(1e-4, (gt_mask | pred_mask).sum()))
		iou_per_class.append((gt_mask & pred_mask).sum() / (gt_mask | pred_mask).sum())
		
	return np.array(iou_per_class).mean()
	
ds_info = loadmat(join(HOME_DIR, 'dataset_info.mat'))
num_class = ds_info['num_labels']-1
num_img = ds_info['num_'+imset.lower()]
ious = []

for idx in range(1, num_img+1):
	gt = loadmat(join(GT_DIR, '%s_%06d_pixeltruth.mat' % (imset, idx)))['truth_img']
	pred = loadmat(join(PRED_DIR, '%s_%06d_prediction.mat' % (imset, idx)))['pred_img']
	ious.append(iou(pred, gt, num_class))
	
	
f = open('iou_%s_sorted.txt' % imset, 'w')
for idx in np.argsort(np.array(ious)):
	img_iou = ious[idx]
	f.write('%d: %f\n' % (idx+1, img_iou))
f.close()