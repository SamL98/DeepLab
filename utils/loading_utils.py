from os.path import join, isfile, isdir
import os

from skimage.io import imread
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

def load_gt(imset, idx, reshape=False):
	ds_path = dsutil.ds_path

	fname = join(ds_path, 'truth', imset, imset+'_%06d_pixeltruth.mat') % idx
	if not isfile(fname):
		return None

	gt = loadmat(fname)['truth_img']

	if reshape:
		gt = gt.ravel()

	return gt

def load_logits(imset, idx, reshape=False):
	ds_path = dsutil.ds_path
	nc = dsutil.nc

	fname = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_logits.mat') % idx
	if not isfile(fname):
		return None

	lgts = loadmat(fname)['logits_img']

	if reshape:
		lgts = lgts.reshape(-1, nc+1)

	return lgts

def load_logit_gt_pair(imset, idx, reshape=True, masked=True, ret_shape=False, ret_mask=False):
	logits = load_logits(imset, idx, reshape=reshape)
	gt = load_gt(imset, idx, reshape=ret_shape)

	if ret_shape:
		shape = gt.shape
		if reshape: gt = gt.ravel()
	elif reshape:
		gt = gt.ravel()

	if masked:
		fg_mask = maskutil.fg_mask_for(gt)
		logits = logits[fg_mask]
		gt = gt[fg_mask]

	gt_info = gt

	if ret_shape:
		gt_info = (gt, shape)

	if ret_mask and masked:
		gt_info = gt_info + (fg_mask,)

	return logits, gt_info

def load_logit_pred_gt_triplet(imset, idx, reshape=True, masked=True, ret_shape=False, ret_mask=False):
	logits, gt_info = load_logit_gt_pair(imset, idx, reshape, masked, ret_shape, ret_mask)
	pred = np.argmax(logits[...,1:], -1) + 1
	return logits, pred, gt_info
	
def load_dl_pred(imset, idx):
	global ds_path
	
	pred_fname = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_prediction.mat') % idx
	if isfile(pred_fname):
		return loadmat(pred_fname)['pred_img']-1
	
	return None
	
def load_calib_pred(imset, idx, name, slc=None, conf=None):
	global ds_path
	
	fname = join(ds_path, 'deeplab_prediction', imset, name, imset+'_%06d_calib_pred.mat') % idx
	if not isfile(fname):
		return None
		
	calib_pred = loadmat(fname)
	masks = calib_pred['masks']
	conf_maps = calib_pred['conf_maps']

	if slc:
		return masks[slc], conf_maps[slc]

	if conf:
		return vizutil.confident_masks(masks, conf_maps, conf)

	return masks, conf_maps

def save_calib_pred(imset, idx, name, confident_masks, confidence_maps):
	global ds_path

	pred_dir = join(ds_path, 'deeplab_prediction', imset, name)
	if not isdir(pred_dir):
		os.mkdir(pred_dir)

	mats = {
		'masks': confident_masks,
		'conf_maps': confidence_maps
	}

	savemat(join(pred_dir, imset+'_%06d_calib_pred.mat') % idx, mats)
