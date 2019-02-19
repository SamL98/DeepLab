import pickle
from collections import namedtuple
import numpy as np
from hdf5storage import loadmat
from os.path import join

Node = namedtuple('Node', 'name node_idx terminals corr_hist count_hist acc_hist')

ds_path = 'D:/datasets/processed/voc2012'
ds_info = loadmat(join(ds_path, 'dataset_info.mat'))
nc = ds_info['num_classes']
classes = ds_info['class_labels']
img_size = 512




'''
Loading Utilities
'''

def num_img_for(imset):
	val_size = 350

	if imset == 'val':
		return val_size
	else:
		return 1449-val_size

def load_gt(imset, idx, reshape=False):
	global ds_path

	gt = loadmat(join(ds_path, imset, imset+'_%06d_pixeltruth.mat') % idx)['truth_img']

	if reshape:
		gt = gt.ravel()

	return gt

def load_logits(imset, idx, reshape=False):
	global ds_path

	lgts = loadmat(join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_logits.mat') % idx)['logits_img']

	if reshape:
		global nc
		lgts = lgts.reshape(-1, nc)

	return lgts



'''
Mask Processing Utilities
'''

def fg_mask_for(gt):
	return ((gt > 0) & (gt < 255))

def sm_of_logits(logits, start_idx=0, zero_pad=False):
	exp_logits = np.exp(logits[:,start_idx:])
	sm = exp_logits / np.maximum(1e-7, exp_logits.sum(-1)[:,np.newaxis])
	
	if zero_pad:
		zero_vec = np.zeros((len(sm)), dtype=sm.dtype)[:,np.newaxis]
		sm = np.concatenate((zero_vec, sm), axis=1)

	return sm



'''
Tree Utilities
'''

def get_depth_of_label(pred_label, slices):
	if pred_label < len(slices[0]):
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
		total_nodes = 0
		for i, slc in enumerate(slices):
			if pred_label > len(slc)+total_nodes:
				total_nodes += len(slc)
				continue
			return len(slices)-i			

def is_in_gt_path(pred_label, gt_label, slice):
	total_nodes = 0
	for slc in slices:
		if pred_label > len(slc)+total_nodes:
			total_nodes += len(slc)
			continue

		gt_remapped = remap_gt(gt_label, slc) + total_nodes
		return gt_remapped == pred_label



'''
Slice Utilities
'''

def read_slices(fname, reset=False):
	with open(fname, 'rb') as f:
		slices = pickle.load(f)

	if reset:
		for slc in slices:
			for node in slc:
				node.corr_hist[:] = 0
				node.count_hist[:] = 0
				node.acc_hist[:] = 0
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

def remap_gt(true_label, slc):
	"""
	Remaps a ground truth terminal label to a truth label within a slice

	:param true_label: terminal label in ground truth
	:param slc: list of clusters
	"""
	for i, node in enumerate(slc):
		if true_label in node.terminals: return i
		
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