import pickle
import numpy as np
from hdf5storage import loadmat, savemat
import multiprocessing as mp
from contextlib import contextmanager
from skimage.io import imread
from os.path import join, isfile, isdir
import os

from node import Node

ds_path = 'D:/datasets/processed/voc2012'
if 'DS_PATH' in os.environ:
	ds_path = os.environ['DS_PATH']

ds_info_fname = join(ds_path, 'dataset_info.mat')
if isfile(ds_info_fname):
	ds_info = loadmat(ds_info_fname)
	classes = ds_info['class_labels'][:-1]
	nc = len(classes)-1


'''
Loading Utilities
'''

def num_img_for(imset):
	val_size = 724
	if imset == 'val':
		return val_size
	else:
		return 1449-val_size
		
def load_rgb(imset, idx):
	global ds_path
	
	return imread(join(ds_path, 'rgb', imset, imset+'_%06d_rgb.jpg' % idx))

def load_gt(imset, idx, reshape=False):
	global ds_path

	gt = loadmat(join(ds_path, 'truth', imset, imset+'_%06d_pixeltruth.mat') % idx)['truth_img']

	if reshape:
		gt = gt.ravel()

	return gt

def load_logits(imset, idx, reshape=False):
	global ds_path

	lgts = loadmat(join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_logits.mat') % idx)['logits_img']

	if reshape:
		global nc
		lgts = lgts.reshape(-1, nc+1)

	return lgts
	
def load_dl_pred(imset, idx):
	global ds_path
	
	return loadmat(join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_prediction.mat') % idx)['pred_img']
	
def load_calib_pred(imset, idx, conf, name):
	global ds_path
	
	fname = join(ds_path, 'deeplab_prediction', imset, name, imset+'_%06d_calib_pred_%.2f.mat') % (idx, conf)
	if not isfile(fname):
		return None
		
	return loadmat(fname)['pred_img']

def save_calib_pred(imset, idx, pred, conf, name):
	global ds_path

	pred_dir = join(ds_path, 'deeplab_prediction', imset, name)
	if not isdir(pred_dir):
		os.mkdir(pred_dir)

	savemat(join(pred_dir, imset+'_%06d_calib_pred_%.2f.mat') % (idx, conf), {'pred_img': pred})
	

'''
Mask Processing Utilities
'''

def fg_mask_for(gt):
	return ((gt > 0) & (gt < 255))

def sm_of_logits(logits, start_idx=0, zero_pad=False):
	logits = logits[...,start_idx:]
	
	logits_max = logits.max(-1)
	if len(logits.shape) > 1:
		logits_max = logits_max[:,np.newaxis]
	
	logits -= logits_max
	exp_logits = np.exp(logits)
	
	exp_logits_sum = exp_logits.sum(-1)
	if len(logits.shape) > 1:
		exp_logits_sum = exp_logits_sum[:,np.newaxis]
		
	sm = exp_logits / np.maximum(1e-7, exp_logits_sum)
	
	if zero_pad:
		zero_vec = np.zeros((len(sm)), dtype=sm.dtype)[:,np.newaxis]
		sm = np.concatenate((zero_vec, sm), axis=1)

	return sm

def get_depth_of_label(pred_label, slices):
	if pred_label <= len(slices[0]):
		# If pred_label is a terminal, it's parent could be a few levels up in the hierarchy.
		#
		# Therefore, iterate through the slices until the terminal label is in a node with more than one child.
		# Then return that depth plus one since the depth of the terminal is actually one lower.
		for i, slc in enumerate(slices):
			for node in slc:
				terms = node.terminals
				if pred_label in terms and len(terms) > 1:
					return len(slices)-i+1
	else:
		# Otherwise, iterate through the slices until the predicted label is within the current slice and return that depth.
		total_nodes = 0
		for i, slc in enumerate(slices):
			if pred_label <= len(slc) + total_nodes:
				return len(slices)-i

			total_nodes += len(slc)			

def is_in_gt_path(pred_label, gt_label, slices):
	total_nodes = 0
	for slc in slices:
		# Accumulate the total nodes before the current slice so that when gt_label is remapped
		# to the local indices of the slice, that base is added to test for equality with the predicted label.
		if pred_label <= len(slc) + total_nodes:
			gt_remapped = remap_gt(gt_label, slc) + total_nodes
			return gt_remapped == pred_label
		
		total_nodes += len(slc)



'''
Slice Utilities
'''

def read_slices(fname, reset=False):
	with open(fname, 'rb') as f:
		slices = pickle.load(f)

	if reset:
		for slc in slices:
			for node in slc:
				node.__init__(node.name, node.node_idx, node.terminals, is_main=True)
	return slices

def save_slices(fname, slices):
	with open(fname, 'wb') as f:
		pickle.dump(slices, f)

def confidence_for_node(vec, node):
	"""
	Takes a logit vector and returns the sum of logit values for terminals in the given node

	:param logit_vec: A length-nc logit vector
	:param node: A node named tuple
	"""
	return vec[node.terminals].sum()

def remap_label(true_label, slc, push_down=False):
	"""
	Remaps a ground truth terminal label to a truth label within a slice

	:param true_label: terminal label in ground truth
	:param slc: list of clusters
	"""
	for i, node in enumerate(slc):
		if true_label in node.terminals:
			lab = i
			if push_down and len(node.terminals) == 1:
				lab = node.terminals[0]
			return lab
		
def remap_scores(vec, slc):
	"""
	Takes a logit or softmax vector and groups the logits by clusters in the slice

	:param vec: A length-nc score vector
	:param slc: A list of clusters
	"""

	conf = []
	for node in slc:
		conf.append(confidence_for_node(vec, node))
	return np.array(conf)



'''
Multiprocessing Utilities
'''

@contextmanager
def poolcontext(num_proc):
    pool = mp.Pool(num_proc)
    yield pool
    pool.terminate()

def get_param_batches(slices, args):
	idx_ordering = None
	idx_ordering_fname = args.imset.lower() + '_ordered.txt'

	if not isfile(idx_ordering_fname):
		from order_by_num_fg import order_imset_by_num_fg
		idx_ordering = order_imset_by_num_fg(args.imset, save=True)
	else:
		with open(idx_ordering_fname) as f:
			idx_ordering = [int(idx) for idx in f.read().split('\n')]

	idx_ordering = np.array(idx_ordering)
	param_batches = []

	for procno in range(args.num_proc):
		idx_batch = idx_ordering[procno::args.num_proc]
		param_batches.append((idx_batch, slices.copy(), args))

	return param_batches
	