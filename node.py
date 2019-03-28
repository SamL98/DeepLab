import os
from os.path import join, isfile, getsize
import numpy as np
from hdf5storage import loadmat, savemat
from scipy.stats import norm
from enum import Enum
import sys

'''
Statistics Utilities
'''
def conf_ints(acc_hist, count_hist, alpha):
	mask = count_hist > 0
	ranges = np.zeros_like(acc_hist)

	p, n = acc_hist[mask], count_hist[mask]
	
	z = norm.ppf(1 - alpha/2)
	pq = p * (1 - p)
	zn = z**2 / (4*n)
	
	if (pq < 0).sum() > 0:
		sys.stdout.write('acc_hist: ' + acc_hist.__repr__() + '\n')
		sys.stdout.flush()
		exit()

	conf_range = z * np.sqrt((pq + zn) / n) / (1 + zn*4)
	new_p = (p + zn*2) / (1 + zn*4)

	conf_range = np.clip(conf_range, 0, new_p)
	conf_range = np.minimum(conf_range, 1-new_p)

	acc_hist[mask] = new_p
	ranges[mask] = conf_range

	return acc_hist, ranges

def parzen_estimate(confs, bins, sigma):
	parzen = np.zeros_like(bins)
	for i, bn in enumerate(bins):
		z = (confs - bn) / sigma
		parzen[i] = (np.exp(-z**2 / 2)).sum()
	return parzen

class node_data_keys(Enum):
	C_HIST = 'c_hist'
	IC_HIST = 'ic_hist'
	TOT_HIST = 'tot_hist'
	ACC_HIST = 'acc_hist'
	INT_RANGES = 'int_ranges'
	

class Node(object):
	def __init__(self, name, node_idx, terminals, data_dir='calib_data', is_main=False):
		self.uid = '%d-%s' % (node_idx, name)
		self.name = name
		self.node_idx = node_idx
		self.terminals = terminals
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
		self.acc_hist = self.node_data[node_data_keys.ACC_HIST.value]
		self.c_hist = self.node_data[node_data_keys.C_HIST.value]
		self.ic_hist = self.node_data[node_data_keys.IC_HIST.value]
		self.tot_hist = self.node_data[node_data_keys.TOT_HIST.value]
		self.int_ranges = self.node_data[node_data_keys.INT_RANGES.value]
		
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
		c_pdf = parzen_estimate(confs[correct_mask], bins, sigma)
		ic_pdf = parzen_estimate(confs[(1-correct_mask).astype(np.bool)], bins, sigma)
		self._accum_stats(c_pdf, ic_pdf, correct_mask.sum(), len(confs) - correct_mask.sum())
	
	def accum_node(self, node):
		self._accum_stats(node.c_pdf, node.ic_pdf, node.n_c, node.n_ic)
			
	def generate_acc_hist(self, nb, alpha):
		attrs = ['c_pdf', 'ic_pdf', 'n_c', 'n_ic']
		for attr in attrs:
			if not hasattr(self, attr): return
		
		self.c_hist = np.round(self.c_pdf / np.maximum(1e-7, self.c_pdf.sum()) * self.n_c)
		self.ic_hist = np.round(self.ic_pdf / np.maximum(1e-7, self.ic_pdf.sum()) * self.n_ic)
		self.tot_hist = self.c_hist + self.ic_hist
		self.tot_hist[self.tot_hist == 0] = 1

		acc_hist = self.c_hist.astype(np.float32) / np.maximum(1e-7, self.tot_hist.astype(np.float32))
		acc_hist = np.minimum(1, acc_hist)
		acc_hist, int_ranges = conf_ints(acc_hist, self.tot_hist, alpha)

		self.acc_hist = acc_hist
		self.int_ranges = int_ranges

		self.node_data = {
			node_data_keys.ACC_HIST.value: self.acc_hist,
			node_data_keys.C_HIST.value: self.c_hist,
			node_data_keys.IC_HIST.value: self.ic_hist,
			node_data_keys.TOT_HIST.value: self.tot_hist,
			node_data_keys.INT_RANGES.value: self.int_ranges
		}

		sys.stdout.write('Saving %s data\n' % self.name)
		sys.stdout.flush()

		savemat(self.node_data_fname, self.node_data)

	def get_conf_acc_hist(self):
		if not hasattr(self, 'node_data'):
			assert isfile(join(self.node_data_fname))
			self.load_node_data()

		return self.acc_hist - self.int_ranges
			
	def get_conf_for_score(self, score):		
		if not hasattr(self, 'node_data'):
			assert isfile(join(self.node_data_fname))
			self.load_node_data()

		lbound_acc_hist = self.get_conf_acc_hist()
		return np.interp(score, np.linspace(0, 1, num=len(lbound_acc_hist)), lbound_acc_hist)

	def reset(self, nb):
		print('Resetting %s node data' % self.name)

		self.remove_attr_if_exists('c_pdf')
		self.remove_attr_if_exists('ic_pdf')

		if hasattr(self, 'node_data_fname') and isfile(self.node_data_fname):
			os.remove(self.node_data_fname)
