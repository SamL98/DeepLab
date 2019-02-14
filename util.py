import pickle
from collections import namedtuple
import numpy as np

Cluster = namedtuple('Cluster', 'name cluster_idx terminals corr_hist count_hist acc_hist')

def read_slices(fname):
	with open(fname, 'rb') as f:
		slices = pickle.load(f)
	return slices

def save_slices(fname, slices):
	with open(fname, 'wb') as f:
		pickle.dump(slices, f)

def confidence_for_cluster(logit_vec, cluster):
	"""
	Takes a logit vector and returns the sum of logit values for terminals in the given cluster

	:param logit_vec: A length-nc logit vector
	:param cluster: A cluster named tuple
	"""
	return logit_vec[cluster.terminals].sum()

def remap_gt(true_label, slc):
	"""
	Remaps a ground truth terminal label to a truth label within a slice

	:param true_label: terminal label in ground truth
	:param slc: list of clusters
	"""
	for i, cluster in enumerate(slc):
		if true_label in cluster.terminals: return i
		
def remap_logits(logit_vec, slc):
	"""
	Takes a terminal logit vector and groups the logits by clusters in the slice

	:param logit_vec: A length-nc logit vector
	:param slc: A list of clusters
	"""

	conf = []
	for cluster in slc:
		conf.append(confidence_for_cluster(logit_vec, cluster))
	return np.array(conf)