import pickle
import numpy as np
from hdf5storage import loadmat, savemat
from skimage.io import imread
from os.path import join, isfile, getsize
import os

class Node(object):
	def __init__(self, name, node_idx, terminals, data_dir='calib_data', is_main=False):
		self.name = name
		self.node_idx = node_idx
		self.terminals = terminals
		self.data_dir = data_dir

		if is_main:
			pid = ''
		else:
			pid = '_' + str(os.getpid())
		self.conf_file = join(self.data_dir, self.name + '_confs' + pid + '.txt')
		self.corr_file = join(self.data_dir, self.name + '_corr' + pid + '.txt')

	def append_confs(self, confs, correct_mask):
		assert confs.shape[0] == correct_mask.shape[0]

		with open(self.conf_file, 'a') as f:
			np.savetxt(f, confs)

		with open(self.corr_file, 'a') as f:
			np.savetxt(f, correct_mask.astype(np.bool))

	def generate_equalized_acc_hist(self, nb):
		assert isfile(self.conf_file)
		assert isfile(self.corr_file)

		confs = np.genfromtxt(self.conf_file)
		correct_mask = np.genfromtxt(self.corr_file).astype(np.bool)

		conf_hist, bins = np.histogram(confs, nb)
		cdf = np.cumsum(confs)

		confs_equa = np.interp(confs, bins[:-1], cdf/cdf[-1])
		_, bins = np.histogram(confs_equa, nb)
		self.bin_edges = bins[:]

		corr_hist = np.zeros((len(bins)-1), dtype=np.float32)
		count_hist = np.zeros_like(corr_hist)

		for i, bin_edge in enumerate(self.bin_edges[1:]):
			bin_mask = confs <= bin_edge
			
			num_pix = bin_mask.sum()
			num_corr = correct_mask[bin_mask].sum()

			corr_hist[i] += num_corr
			count_hist[i] += num_pix

		self.acc_hist = corr_hist.astype(np.float32) / np.maximum(1e-7, count_hist.astype(np.float32))

	def get_file_contents(self):
		if not (isfile(self.conf_file) and isfile(self.corr_file)):
			return None, None
			
		if getsize(self.conf_file) == 0 or getsize(self.corr_file) == 0:
			return None, None
			
		return np.genfromtxt(self.conf_file), np.genfromtxt(self.corr_file).astype(np.bool)
		
	def reset(self):
		if isfile(self.conf_file):
			os.remove(self.conf_file)
		
		if isfile(self.corr_file):
			os.remove(self.corr_file)
		

ds_path = 'D:/datasets/processed/voc2012'
if 'DS_PATH' in os.environ:
	ds_path = os.environ['DS_PATH']

ds_info = loadmat(join(ds_path, 'dataset_info.mat'))
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
	
def load_calib_pred(imset, idx, conf):
	global ds_path
	
	fname = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_calib_pred_%.2f.mat') % (idx, conf)
	if not isfile(fname):
		return None
		
	return loadmat(fname)['pred_img']

def save_calib_pred(imset, idx, pred, conf):
	global ds_path
	
	savemat(join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_calib_pred_%.2f.mat') % (idx, conf), {'pred_img': pred})
	

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
		# Otherwise, iterate through the slices until the predicted label is within the current slice and return that depth.
		total_nodes = 0
		for i, slc in enumerate(slices):
			if pred_label > len(slc)+total_nodes+1:
				total_nodes += len(slc)
				continue
			return len(slices)-i			

def is_in_gt_path(pred_label, gt_label, slices):
	total_nodes = 0
	for slc in slices:
		# Accumulate the total nodes before the current slice so that when gt_label is remapped
		# to the local indices of the slice, that base is added to test for equality with the predicted label.
		if pred_label > len(slc)+total_nodes+1:
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
				node.reset()
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