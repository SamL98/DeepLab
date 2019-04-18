import os
from os.path import join, isfile, getsize
import numpy as np
from hdf5storage import loadmat, savemat
from enum import Enum
import sys

class node_data_keys(Enum):
	CDF = 'cdf'
	LL_HIST = 'll_hist'

class Node(object):
	def __init__(self, name, node_idx, children, data_dir='calib_data', is_main=False):
		self.uid = '%d-%s' % (node_idx, name)
		self.name = name
		self.node_idx = node_idx
		self.children = children
		self.data_dir = data_dir

		if is_main:
			self.node_data_fname = join(self.data_dir, '%s_node_data.mat' % self.uid)
			if isfile(self.node_data_fname):
				self.load_node_data()

	def add_attr_if_not_exists(self, attr_name, attr_val):
		if not hasattr(self, attr_name):
			setattr(self, attr_name, attr_val)

	def remove_attr_if_exists(self, attr_name):
		if hasattr(self, attr_name):
			delattr(self, attr_name)

	def load_node_data(self):
		self.node_data = loadmat(self.node_data_fname)
		self.cdf = self.node_data[node_data_keys.CDF.value]
		self.ll_hist = self.node_data[node_data_keys.LL_HIST.value]
		
	def _accum_stats(self, cdf):
		if not hasattr(self, 'cdf'):
			self.cdf = cdf
		else:
			self.cdf = np.concatenate((self.cdf, self.cdf[-1] + cdf))

	def accum_scores(self, confs):
		self._accum_stats(confs.cumsum())
	
	def accum_node(self, node):
		self._accum_stats(node.cdf)
			
	def generate_ll_hist(self, nb):
		if not hasattr(self, 'cdf'): return
		
		norm_cdf = self.cdf / self.cdf[-1]
		bins = np.floor(np.linspace(0, len(norm_cdf)-1, num=nb)).astype(np.uint16)
		self.ll_hist = norm_cdf[bins]

		self.node_data = {
			node_data_keys.LL_HIST.value: self.ll_hist,
			node_data_keys.CDF.value: self.cdf
		}

		savemat(self.node_data_fname, self.node_data)

	def conf_for_scores(self, scores):
		if not hasattr(self, 'node_data'):
			assert isfile(join(self.node_data_fname))
			self.load_node_data()

		nb = len(self.ll_hist)
		res = 1./nb

		binvec = np.minimum(np.floor(scores/res).astype(np.uint16), nb-1)
		return self.ll_hist[binvec]

	def reset(self):
		self.remove_attr_if_exists('cdf')

		if hasattr(self, 'node_data_fname') and isfile(self.node_data_fname):
			os.remove(self.node_data_fname)
