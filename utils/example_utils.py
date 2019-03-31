import numpy as np
import struct as st
from os.path import join, isdir
import os
import atexit
from functools import reduce

import ds_utils as dsutil
import mask_utils as maskutil

data_dir = join(dsutil.ds_path, 'flat_dataset')
if not isdir(data_dir):
	os.mkdir(data_dir)

GT = 'gt'
LOGITS = 'logits'
SHAPE = 'shape'

DTYPES = {
	GT: np.uint8,
	LOGITS: np.float64,
	SHAPE: np.uint32
}

INT_SIZE = 4
ST_SHAPE_FMT = 'II'

SHAPE_F = 'shapes'
FG_F = 'fg_masks'
LGT_F = 'logits'
GT_F = 'ground_truth_masks'

f_shape = None
f_fg = None
f_lgt = None
f_gt = None

files = {
	SHAPE_F: f_shape,
	FG_F: f_fg, 
	LGT_F: f_lgt,
	GT_F: f_gt
}

def close_files():
	map(lambda kv: kv[1].close(), files.items())

atexit.register(close_files)

def serialize_lgt_gt_pair(logits, gt, imset, chunkno):
	assert len(logits) == len(gt)
	assert logits.dtype == DTYPES[LOGITS]
	assert gt.dtype == DTYPES[GT]
	
	chnk = str(chunkno)

	for fname, fs in files.items():
		if fs is None or not chnk in fs: 
			files[fname] = {
				chnk: open(join(data_dir, f'{imset}_{fname}-{chnk}.txt'), 'ab')
			}

	h, w = gt.shape
	fg_mask = maskutil.fg_mask_for(gt)
	gt = gt[fg_mask]
	logits = logits[fg_mask]

	files[SHAPE_F][chnk].write(st.pack(ST_SHAPE_FMT, h, w))
	files[FG_F][chnk].write(fg_mask.tobytes())
	files[LGT_F][chnk].write(logits[...,1:].tobytes())
	files[GT_F][chnk].write(gt.tobytes())

def unserialize_examples(imset, n_ex, chunkno):
	chnk = str(chunkno)

	for fname, fs in files.items():
		if fs is None or not chnk in fs: 
			files[fname] = {
				chnk: open(join(data_dir, f'{imset}_{fname}-{chnk}.txt'), 'rb')
			}

	hws = np.fromstring(files[SHAPE_F][chnk].read(2 * n_ex * INT_SIZE), dtype=DTYPES[SHAPE])
	if len(hws) != 2*n_ex:
		n_ex = len(hws)//2

	num_fg_bytes = reduce(lambda x,y: x*y, hws)	
	fg_masks = np.fromstring(files[FG_F][chnk].read(num_fg_bytes))
	num_fg_pix = fg_masks.sum()

	num_lgt_bytes = np.dtype(DTYPES[LOGITS]).itemsize * num_fg_pix * dsutil.nc
	num_gt_bytes = np.dtype(DTYPES[GT]).itemsize * num_fg_pix

	lgts = np.fromstring(files[LGT_F][chnk].read(num_lgt_bytes))	
	gts = np.fromstring(files[GT_F][chnk].read(num_gt_bytes))

	h_col = np.expand_dims(hws[::2], 0)
	w_col = np.expand_dims(hws[1::2], 0)
	shapes = np.concatenate((h_col, w_col), 1) 
	num_pix = (h_col * w_col).ravel()

	return n_ex, shapes, num_pix, fg_masks, lgts, gts
