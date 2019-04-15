import os
from os.path import join, isfile
import numpy as np
from hdf5storage import loadmat, savemat
from enum import Enum
import sys

class node_data_keys(Enum):
	C_HIST = 'c_hist'
	IC_HIST = 'ic_hist'

class Node(object):
	def __init__(self, name, node_idx, children, nb, data_dir='calib_data', is_main=False):
		self.uid = '%d-%s' % (node_idx, name)
		self.name = name
		self.node_idx = node_idx
		self.children = children
		self.data_dir = data_dir

		self.nb = nb
		self.res = 1./nb

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
		self.c_hist = self.node_data[node_data_keys.C_HIST.value]
		self.ic_hist = self.node_data[node_data_keys.IC_HIST.value]

	def _bin_sm(self, sm):
		binvec = np.floor(sm/self.res).astype(np.uint16)
		return np.minimum(binvec, self.nb-1)

	def accum_sm(self, min_sm, max_sm, correct_mask):
		self.add_attr_if_not_exists('c_hist', np.zeros(self.nb, dtype=np.uint64))
		self.add_attr_if_not_exists('ic_hist', np.zeros(self.nb, dtype=np.uint64))

		min_bins = self._bin_sm(min_sm)
		max_bins = self._bin_sm(max_sm)

		corr_min_bins = min_bins[correct_mask]
		corr_max_bins = max_bins[correct_mask]

		for min_bin, max_bin in zip(corr_min_bins, corr_max_bins):
			self.c_hist[min_bin, max_bin] += 1

		incorrect_mask = (1-correct_mask).astype(np.bool)
		incorr_min_bins = min_bins[incorrect_mask]
		incorr_max_bins = max_bins[incorrect_mask]

		for min_bin, max_bin in zip(incorr_min_bins, incorr_max_bins):
			self.ic_hist[min_bin, max_bin] += 1
	
	def accum_node(self, node):
		if hasattr(node, 'c_hist'):
			self.add_attr_if_not_exists('c_hist', np.zeros_like(node.c_hist))
			self.c_hist += node.c_hist

		if hasattr(node, 'ic_hist'):
			self.add_attr_if_not_exists('ic_hist', np.zeros_like(node.ic_hist))
			self.ic_hist += node.ic_hist
			
	def save(self):
		self.node_data = {
			node_data_keys.C_HIST.value: self.c_hist,
			node_data_keys.IC_HIST.value: self.ic_hist,
		}
		savemat(self.node_data_fname, self.node_data)

	def confs_for_sm(self, min_sm, max_sm):
		if not hasattr(self, 'node_data'):
			assert isfile(join(self.node_data_fname))
			self.load_node_data()

		min_bins = self._bin_sm(min_sm)
		max_bins = self._bin_sm(max_sm)

		confs = np.zeros_like(min_sm)

		for i, (min_bin, max_bin) in enumerate(zip(min_bins, max_bins)):
			num_c = self.c_hist[min_bin:max_bin].sum()
			num_ic = self.ic_hist[min_bin:max_bin].sum()
			confs[i] = num_c / max(1e-7, float(num_ic))

		return confs

	def reset(self):
		self.remove_attr_if_exists('c_pdf')
		self.remove_attr_if_exists('ic_pdf')

		if hasattr(self, 'node_data_fname') and isfile(self.node_data_fname):
			os.remove(self.node_data_fname)
