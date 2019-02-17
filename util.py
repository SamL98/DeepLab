import pickle
from collections import namedtuple
import numpy as np
from hdf5storage import loadmat
from os.path import join

Node = namedtuple('Cluster', 'name cluster_idx terminals corr_hist count_hist acc_hist')

ds_path = 'D:/datasets/processed/voc2012'
ds_info = loadmat(join(ds_path, 'dataset_info.mat'))
nc = ds_info['num_classes']
classes = ds_info['class_labels']
img_size = 512

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

def fg_mask_for(gt):
	return ((gt > 0) & (gt < 255))

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

def confidence_for_cluster(vec, cluster):
	"""
	Takes a logit vector and returns the sum of logit values for terminals in the given cluster

	:param logit_vec: A length-nc logit vector
	:param cluster: A cluster named tuple
	"""
	return vec[cluster.terminals].sum()

def remap_gt(true_label, slc):
	"""
	Remaps a ground truth terminal label to a truth label within a slice

	:param true_label: terminal label in ground truth
	:param slc: list of clusters
	"""
	for i, cluster in enumerate(slc):
		if true_label in cluster.terminals: return i
		
def remap_scores(vec, slc):
	"""
	Takes a logit or softmax vector and groups the logits by clusters in the slice

	:param vec: A length-nc score vector
	:param slc: A list of clusters
	"""

	conf = []
	for cluster in slc:
		conf.append(confidence_for_cluster(vec, cluster))
	return np.array(conf)