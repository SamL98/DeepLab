import os
from os.path import join, isfile, getsize
import numpy as np
from hdf5storage import loadmat, savemat
from enum import Enum
import sys

class node_data_keys(Enum):
	C_HIST = 'c_hist'
	IC_HIST = 'ic_hist'
	

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
		self.c_hist = self.node_data[node_data_keys.C_HIST.value]
		self.ic_hist = self.node_data[node_data_keys.IC_HIST.value]
		
	def _accum_stats(self, c_pdf, ic_pdf, n_c, n_ic):
		self.add_attr_if_not_exists('c_pdf', np.zeros_like(c_pdf))
		self.add_attr_if_not_exists('ic_pdf', np.zeros_like(ic_pdf))
		self.add_attr_if_not_exists('n_c', 0)
		self.add_attr_if_not_exists('n_ic', 0)

		self.c_pdf += c_pdf
		self.ic_pdf += ic_pdf
		self.n_c += n_c
		self.n_ic += n_ic

	def accum_scores(self, confs, correct_mask, nb, sigma):
		bins = np.linspace(0, 1, num=nb)
		c_pdf = util.parzen_estimate(confs[correct_mask], bins, sigma)
		ic_pdf = util.parzen_estimate(confs[(1-correct_mask).astype(np.bool)], bins, sigma)
		self._accum_stats(c_pdf, ic_pdf, correct_mask.sum(), len(confs) - correct_mask.sum())
	
	def accum_node(self, node):
		self._accum_stats(node.c_pdf, node.ic_pdf, node.n_c, node.n_ic)
			
	def generate_counts(self):
		attrs = ['c_pdf', 'ic_pdf', 'n_c', 'n_ic']
		for attr in attrs:
			if not hasattr(self, attr): return
		
		self.c_hist = np.round(self.c_pdf / np.maximum(1e-7, self.c_pdf.sum()) * self.n_c)
		self.ic_hist = np.round(self.ic_pdf / np.maximum(1e-7, self.ic_pdf.sum()) * self.n_ic)

		self.node_data = {
			node_data_keys.C_HIST.value: self.c_hist,
			node_data_keys.IC_HIST.value: self.ic_hist,
		}

		savemat(self.node_data_fname, self.node_data)

	def reset(self, nb):
		print('Resetting %s node data' % self.name)

		self.remove_attr_if_exists('c_pdf')
		self.remove_attr_if_exists('ic_pdf')

		if hasattr(self, 'node_data_fname') and isfile(self.node_data_fname):
			os.remove(self.node_data_fname)
