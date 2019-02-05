import os
from os.path import join

import h5py
import numpy as np
from PIL import Image

from deeplab.utils import save_annotation

import sys

model_ts = int(sys.argv[1])
model_name = 'model-pc-%d' % model_ts

root_dir = join('E:', 'lerner', 'deeplab', 'cache_data', model_name, 'vis')
src_dir = join(root_dir, 'segmentation_results')
dst_dir = join(root_dir, 'segmentation_results_masked')
gt_mask_dir = join('D:', 'datasets', 'processed', 'pascalcontext', 'truth-', 'val')

if not os.path.exists(dst_dir):
	os.mkdir(dst_dir)

for f in os.listdir(src_dir):
	if 'image' in f:
		os.rename(join(src_dir, f), join(dst_dir, f))
		continue

	idx = 1+int(f[:f.index('_')])
	pred = np.array(Image.open(join(src_dir, f)).convert('L'))
	
	with h5py.File(join(gt_mask_dir, 'val_%06d_pixeltruth.mat' % idx)) as mat:
		gt = mat['truth_img'][:].T

	pred[gt==0] = 0

	#im = Image.fromarray(pred).convert('RGB')	
	#im.save(join(dst_dir, f)) 

	save_annotation.save_annotation(pred, dst_dir, f) 
