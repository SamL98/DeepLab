import os
from os.path import join, isfile, getsize
import numpy as np
from hdf5storage import loadmat, savemat
from scipy.stats import norm
from sklearn.neighbors.kde import KernelDensity
from enum import Enum

'''
Statistics Utilities
'''
def conf_ints(pdf, count_hist, alpha):
	mask = count_hist > 0
	ranges = np.zeros_like(pdf)

	p, n = pdf[mask], count_hist[mask]
	
	z = norm.ppf(1 - alpha/2)
	pq = p * (1 - p)
	zn = z**2 / (4*n)

	conf_range = z * np.sqrt((pq_hat + zn) / n) / (1 + zn*4)
	new_p = (p + zn*2) / (1 + zn*4)

	pdf[mask] = new_p
	ranges[mask] = conf_range

	return pdf, ranges

def pdf_for_confs(confs, bins, sigma=0.75):
	kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(confs)
	log_dens = kde.score_samples(bins)
	return np.exp(log_dens)

def count_hist_for_confs(confs, bins):
	nb = len(bins)-1
	res = 1./nb
	count_hist = np.zeros((nb), dtype=np.int32)
	bin_vec = np.maximum(nb-1, np.floor(confs/res).astype(np.uint8))
	return np.histogram(bin_vec, bins=bins)[0]

def density(confs, mask, bins, sigma, alpha):
	n = mask.sum()
	confs_masked = confs[mask]
	pdf = pdf_for_confs(confs_masked, bins, sigma=sigma)
	count_hist = count_hist_for_confs(confs_masked, bins)
	ci = conf_ints(pdf, count_hist, alpha)
	return pdf, ci, n

class node_data_keys(Enum):
	ACC_HIST = 'acc_hist'
	C_HIST = 'c_hist'
	IC_HIST = 'ic_hist'
	COUNT_HIST = 'count_hist'
	INT_RANGES = 'int_ranges'
	

class Node(object):
	def __init__(self, name, node_idx, terminals, data_dir='calib_data', is_main=False):
		self.uid = '%d-%s' % (node_idx, name)
		self.name = name
		self.node_idx = node_idx
		self.terminals = terminals
		self.data_dir = data_dir

		# Since we generally use multiple clones of each node to obtain the calibration data,
		# the pid is appended to the filenames of the node if we are a clone
		if is_main:
			pid = ''
		else:
			pid = '_' + str(os.getpid())

		if is_main:
			self.node_data_fname = join(self.data_dir, '%s_node_data.mat' % self.uid)
			if isfile(self.node_data_fname):
				self.load_node_data()

	def add_attr_if_not_exists(self, attr_name, attr_val):
		if not hasattr(self, attr_name):
			setattr(self, attr_name, attr_val)


	def load_node_data(self):
		self.node_data = loadmat(self.node_data_fname)
		self.acc_hist = self.node_data[node_data_keys.ACC_HIST.value]
		self.int_ranges = self.node_data[node_data_keys.INT_RANGES.value]

		
	def _accum_stats(self, n_c, c_pdf, c_ci, n_ic, ic_pdf, ic_ci):
		add_attr_if_not_exists(self, 'n_c', 0)
		add_attr_if_not_exists(self, 'n_ic', 0)
		add_attr_if_not_exists(self, 'n_pdf', 1)
		add_attr_if_not_exists(self, 'tot_c_pdf', np.zeros_like(c_pdf))
		add_attr_if_not_exists(self, 'tot_ic_pdf', np.zeros_like(c_pdf))
		add_attr_if_not_exists(self, 'tot_c_ci', np.zeros_like(c_ci))
		add_attr_if_not_exists(self, 'tot_ic_ci', np.zeros_like(c_ci))

		self.n_c += n_c
		self.n_ic += n_ic

		self.n_pdf += 1
		self.tot_c_pdf += c_pdf
		self.tot_ic_pdf += ic_pdf

		self.c_pdf = self.tot_c_pdf / self.n_pdf
		self.ic_pdf = self.tot_ic_idf / self.n_pdf

		self.tot_c_ci += c_ci
		self.tot_ic_ci += ic_ci

		self.c_ci = self.tot_c_ci / self.n_pdf
		self.ic_ci = self.tot_ic_ci / self.n_pdf

	def accum_pdfs(self, confs, correct_mask, nb, sigma=0.75, alpha=0.05):
		bins = np.linspace(0, 1, num=nb+1)
		c_pdf, c_ci, n_c = density(confs, correct_mask, bins, sigma, alpha)
		ic_pdf, ic_ci, n_ic = density(confs, 1-correct_mask, bins, sigma, alpha)
		self._accum_stats(n_c, c_pdf, c_ci, n_ic, ic_pdf, ic_ci)
	

	def accum_node(self, node):
		self._accum_stats(node.n_c, node.c_pdf, node.c_ci, node.n_ic, node.ic_pdf, node.ic_ci)

	def _avg_hist(self, v1, v2, n1, n2):
		h1 = np.ceil(v1*n1)
		h2 = np.ceil(v2*n2)
		t = h1 + h2
		return h1 / np.maximum(1e-7, t)
			
	def generate_acc_hist(self, nb):
		attrs = ['n_c', 'n_ic', 'c_pdf', 'ic_pdf', 'c_ci', 'ic_ci']
		for attr in attrs:
			if not hasattr(self, attr): return

		bin_edges = np.linspace(0, 1, num=nb+1)
		
		acc_hist = self._avg_hist(self.c_pdf, self.ic_pdf, self.n_c, self.n_ic)
		int_ranges = self._avg_hist(self.c_ci, self.ic_ci, self.n_c, self.n_ic)

		self.acc_hist = acc_hist
		self.node_data = {
			node_data_keys.ACC_HIST.value: acc_hist,
			node_data_keys.INT_RANGES.value: int_ranges
		}

		savemat(self.node_data_fname, self.node_data)
			
		
	def get_conf_for_score(self, score):		
		if not hasattr(self, 'node_data'):
			assert isfile(join(self.node_data_fname))
			self.load_node_data()

		nb = len(self.acc_hist)
		res = 1./nb
		binno = np.floor(score/res)
		acc_val = self.acc_hist[binno]

		if hasattr(self, 'int_ranges'):
			acc_val -= self.int_ranges[i]

		return acc_val
