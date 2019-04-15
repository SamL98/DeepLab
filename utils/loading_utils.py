from os.path import join, isfile, isdir
import os

from skimage.io import imread, imsave
from hdf5storage import loadmat, savemat

import numpy as np

import mask_utils as maskutil
import ds_utils as dsutil
import viz_utils as vizutil

def load_rgb(imset, idx):
	ds_path = dsutil.ds_path
	fname = join(ds_path, 'rgb', imset, imset+'_%06d_rgb.jpg' % idx)
	if not isfile(fname):
		return None

	return imread(fname)
	
def save_rgb(imset, idx, rgb):
	ds_path = dsutil.ds_path
	imsave(join(ds_path, 'rgb', imset, imset+'_%06d_rgb.jpg' % idx), rgb)

def _rgb_aug_path(imset, idx):
	return join(dsutil.ds_path, 'rgb_aug', imset, imset+'_%06d_rgb.mat' % idx)

def _gt_path(imset, idx):
	return join(dsutil.ds_path, 'Truth', imset, imset+'_%06d_pixeltruth.mat' % idx)

def _lgt_aug_path(imset, idx):
	return join(dsutil.ds_path, 'logits_aug', imset, imset+'_%06d_logits.mat' % idx)

def save_rgb_aug(imset, idx, rgb, flip_idxs):
	savemat(_rgb_aug_path(imset, idx), {'rgb_img': rgb, 'flip_idxs': flip_idxs})

def load_rgb_aug(imset, idx):
	loadmat(_rgb_aug_path(imset, idx))['rgb_img']

def load_gt(imset, idx):
	loadmat(_gt_path(imset, idx))['truth_img']

def save_lgt_aug(imset, idx, lgt):
	savemat(_lgt_aug_path(imset, idx), {'logits_img': lgt})

def load_lgt_aug(imset, idx):
	loadmat(_lgt_aug_path(imset, idx))['logits_img']

def load_calib_pred(imset, idx, name, slc=None, conf=None):
	ds_path = dsutil.ds_path
	
	fname = join(ds_path, 'deeplab_prediction', imset, name, imset+'_%06d_calib_pred.mat' % idx)
	if not isfile(fname):
		return None
		
	calib_pred = loadmat(fname)
	masks = calib_pred['masks']
	conf_maps = calib_pred['conf_maps']

	if slc:
		return masks[slc], conf_maps[slc]

	if conf:
		return vizutil.confident_mask(masks, conf_maps, conf)

	return masks, conf_maps

def save_calib_pred(imset, idx, name, confident_masks, confidence_maps):
	ds_path = dsutil.ds_path

	pred_dir = join(ds_path, 'deeplab_prediction', imset, name)
	#if not isdir(pred_dir):
	#	os.mkdir(pred_dir)

	mats = {
		'masks': confident_masks,
		'conf_maps': confidence_maps
	}

	savemat(join(pred_dir, imset+'_%06d_calib_pred.mat' % idx), mats)
