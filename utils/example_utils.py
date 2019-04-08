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

FG = 'gt'
GT = 'gt'
LOGITS = 'logits'
SHAPE = 'shape'

DTYPES = {
	FG: np.bool,
	GT: np.uint8,
	LOGITS: np.float32,
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

def open_files_for_reading(imset, chnk):
	for fname, fs in files.items():
		chnk_fname = join(data_dir, f'{imset}_{fname}-{chnk}.txt')
		if fs is None:
			files[fname] = { chnk: open(chnk_fname, 'rb') }
		elif not chnk in fs:
			files[fname][chnk] = open(chnk_fname, 'rb')

def read_logits_and_gt(num_pix, chnk, lgts_out, gt_out):
	num_lgt_bytes = np.dtype(DTYPES[LOGITS]).itemsize * num_pix * dsutil.nc
	num_gt_bytes = np.dtype(DTYPES[GT]).itemsize * num_pix

	gts = np.fromstring(files[GT_F][chnk].read(num_gt_bytes), dtype=DTYPES[GT])
	
	done = False
	if len(gts) < num_pix:
		done = True
		num_pix = len(gts)

	gt_out[:num_pix] = gts
	lgts_out[:num_pix] = np.fromstring(files[LGT_F][chnk].read(num_lgt_bytes), dtype=DTYPES[LOGITS]).reshape(-1, dsutil.nc)	

	return done, num_pix

def unserialize_examples_for_calib(imset, n_pix, chunkno, lgts_out, gt_out):
	chnk = str(chunkno)
	open_files_for_reading(imset, chnk)
	return read_logits_and_gt(n_pix, chnk, lgts_out, gt_out)
