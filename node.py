import os
from os.path import join, isfile, getsize
import numpy as np

class Node(object):
	def __init__(self, name, node_idx, terminals, data_dir='calib_data', is_main=False):
		self.uid = '%d-%s' % (node_idx, name)
		self.name = name
		self.node_idx = node_idx
		self.terminals = terminals
		self.data_dir = data_dir

		if is_main:
			pid = ''
		else:
			pid = '_' + str(os.getpid())
		self.conf_file = join(self.data_dir, '%s_confs%s.txt' % (self.uid, pid))
		self.corr_file = join(self.data_dir, '%s_corr%s.txt' % (self.uid, pid))

		if is_main:
			bin_edge_fname = join(self.data_dir, '%s_bin_edges.txt' % self.uid)
			if isfile(bin_edge_fname):
				self.bin_file = bin_edge_fname
				self.bin_edges = np.genfromtxt(bin_edge_fname)

			acc_hist_fname = join(self.data_dir, '%s_acc_hist.txt' % self.uid)
			if isfile(acc_hist_fname):
				self.acc_file = acc_hist_fname
				self.acc_hist = np.genfromtxt(acc_hist_fname)

	def get_fg_count(self):
		assert isfile(self.corr_file) and getsize(self.corr_file) > 0

		return np.genfromtxt(self.corr_file).shape[0]

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
		cdf = np.cumsum(conf_hist)

		confs_equa = np.interp(confs, bins[:-1], cdf/cdf[-1])
		_, bins = np.histogram(confs_equa, nb)
		self.bin_edges = bins[:]

		self.bin_file = join(self.data_dir, '%s_bin_edges.txt' % self.uid)
		np.savetxt(self.bin_file, self.bin_edges)

		corr_hist = np.zeros((len(bins)-1), dtype=np.float32)
		count_hist = np.zeros_like(corr_hist)

		for i, bin_edge in enumerate(self.bin_edges[1:]):
			bin_mask = confs <= bin_edge
			
			num_pix = bin_mask.sum()
			num_corr = correct_mask[bin_mask].sum()

			corr_hist[i] += num_corr
			count_hist[i] += num_pix

		self.acc_hist = corr_hist.astype(np.float32) / np.maximum(1e-7, count_hist.astype(np.float32))

		self.acc_file = join(self.data_dir, '%s_acc_hist.txt' % self.uid)
		np.savetxt(self.acc_file, self.acc_hist)


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