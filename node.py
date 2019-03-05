import os
from os.path import join, isfile, getsize
import numpy as np
from hdf5storage import loadmat, savemat
from scipy.stats import norm, gaussian_kde

'''
Statistics Utilities
'''
def calculate_conf_interval(p_hat, n, alpha):
	if n == 0:
		return p_hat, 0
		
	z = norm.ppf(1 - alpha/2)
	pq_hat = p_hat * (1 - p_hat)
	z_norm = z**2 / (4*n)
	conf_range = z * np.sqrt((pq_hat + z_norm) / n) / (1 + z_norm * 4)
	p_hat_adj = (p_hat + z_norm*2) / (1 + z_norm*4)
	return p_hat_adj, conf_range

def pdf_for_confs(confs, bins):
	pdf = gaussian_kde(confs)
	return pdf(bins)

def hist_for_confs(confs, bins):
	pdf = pdf_for_confs(confs, bins)
	n_c = len(confs)
	return np.ceil(pdf * n_c)

node_data_keys = {
	ACC_HIST: 'acc_hist',
	C_HIST: 'c_hist',
	IC_HIST: 'ic_hist',
	COUNT_HIST: 'count_hist',
	INT_RANGES: 'int_ranges'
}
	

class Node(object):
	def __init__(self, name, node_idx, terminals, data_dir='calib_data', is_main=False):
		'''
		Instantiate a Node object
		
		:param name: the name of the node
		:param node_idx: the index of the node within the hierarchy
		:param terminals: the indices of all terminal labels within this node's subtree
		:param data_dir: the directory to save data files in
		:param is_main: whether or not this node is the only clone for this node
		'''
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

		self.conf_file = join(self.data_dir, '%s_confs%s.txt' % (self.uid, pid))
		self.corr_file = join(self.data_dir, '%s_corr%s.txt' % (self.uid, pid))

		if is_main:
			self.node_data_fname = join(self.data_dir, '%s_node_data.mat' % self.uid)
			if isfile(self.node_data_fname):
				self.load_node_data()


	def load_node_data(self):
		self.node_data = loadmat(self.node_data_fname)
		self.acc_hist = self.node_data[node_data_keys.ACC_HIST]
		self.c_hist = self.node_data[node_data_keys.C_HIST]
		self.ic_hist = self.node_data[node_data_keys.IC_HIST]
		self.count_hist = self.node_data[node_data_keys.COUNT_HIST]
		self.int_ranges = self.node_data[node_data_keys.INT_RANGES]

				
	def get_fg_count(self):
		'''
		Return the number of foreground pixels that the were classified as the current node
		'''
		assert isfile(self.corr_file)

		if getsize(self.corr_file) == 0:
			return 0

		# The number of foreground pixels is equivalent to the number of pixels stored in each file.
		# Load the correct mask file because the data type is smaller and therefore should be a bit faster to load.
		return np.genfromtxt(self.corr_file).shape[0]

		
	def append_confs(self, confs, correct_mask):
		'''
		Write the given confidences and corresponding correct booleans to disk

		:param confs: a float vector of the softmax score
		:param correct_mask: a boolean vector of whether each confs corresponds to a correct prediction
		'''
		assert confs.shape[0] == correct_mask.shape[0], 'confs and correct_mask should have same shape.'

		with open(self.conf_file, 'a') as f:
			np.savetxt(f, confs)

		with open(self.corr_file, 'a') as f:
			np.savetxt(f, correct_mask.astype(np.bool))

			
	def generate_acc_hist(self, nb, slc_len, lb=True, alpha=0.75):
		'''
		Generates the accuracy histogram for the current node.

		:param nb: the number of bins in the histogram
		:param lb: whether or not to store the lower bound of the accuracy in each bin (according to the Wilson confidence interval)
		:param alpha: the confidence for the Wilson confidence interval -- only matters when lb=True
		'''
		assert isfile(self.conf_file), '%s conf file does not exist' % self.uid
		assert isfile(self.corr_file), '%s corr file does not exist' % self.uid
		
		if getsize(self.conf_file) == 0:
			return

		confs, correct_mask = self.get_file_contents()
		if len(confs) == 0:
			return

		#abs_lb = 1./slc_len
		abs_lb = 0
		bin_edges = np.linspace(abs_lb, 1, num=nb+1)

		c_hist = hist_for_confs(confs[correct_mask], bin_edges)
		ic_hist = hist_for_confs(confs[!correct_mask], bin_edges)
		
		t_hist = c_hist + ic_hist
		acc_hist = c_hist.astype(np.float32) / np.maximum(1e-7, t_hist.astype(np.float32))
		count_hist = np.zeros_like(t_hist)

		for i, be in enumerate(bin_edges[1:]):
			mask = (confs > bin_edges[i]) & (confs <= be)
			count_hist[i] = mask.sum()

		if lb:
			int_ranges = []

			for i, (acc_val, bin_count) in enumerate(zip(acc_hist, count_hist)):
				acc_adj, conf_range = calculate_conf_interval(acc_val, bin_count, alpha)
				int_ranges.append(conf_range)
				acc_hist[i] = acc_adj

			int_ranges = np.array(int_ranges)
			self.int_ranges = int_ranges

		self.acc_hist = acc_hist
		self.c_hist = c_hist
		self.ic_hist = ic_hist
		self.count_hist = count_hist
		self.node_data = {
			node_data_keys.ACC_HIST: acc_hist,
			node_data_keys.C_HIST: c_hist,
			node_data_keys.IC_HIST: ic_hist,
			node_data_keys.COUNT_HIST: count_hist,
		}

		if hasattr(self, 'int_ranges'):
			self.node_data[node_data_keys.INT_RANGES] = self.int_ranges

		savemat(self.node_data_fname, self.node_data)
			

	def interp_acc_hist(self, count_hist):
		'''
		Interpolate the accuracy histogram

		:param count_hist: the number of pixels that fell within each bin
		'''
		assert hasattr(self, 'acc_hist'), 'Node object does not have acc hist attribute.'

		count_hist = np.array(count_hist)
		
		# We want to interpolate where the accuracy histogram is 0 using the other bins as the reference points.
		xs = np.argwhere(self.acc_hist == 0).ravel()
		xp = np.argwhere(self.acc_hist > 0).ravel()
		yp = self.acc_hist[xp]
	
		if count_hist[0] == 0 and self.acc_hist[0] == 0:
			xp = np.concatenate(([0], xp))
			yp = np.concatenate(([0], yp))
		
		if count_hist[-1] == 0 and self.acc_hist[-1] == 0:
			xp = np.concatenate((xp, [len(self.acc_hist)-1]))
			yp = np.concatenate((yp, [1]))

		y_interp = np.interp(xs, xp, yp)
		self.acc_hist[self.acc_hist == 0] = y_interp

		
	def get_conf_for_score(self, score):
		'''
		Get the calibrated confidence given a softmax score

		:param score: the softmax score to calibrate
		'''
		assert isfile(self.node_data_fname)
		
		if getsize(self.corr_file) == 0:
			return 0
		
		if not hasattr(self, 'node_data'):
			self.load_node_data()

		nb = len(self.acc_hist)
		res = 1./nb
		binno = np.floor(score/res)
		acc_val = self.acc_hist[binno]

		if hasattr(self, 'int_ranges'):
			acc_val -= self.int_ranges[i]

		return acc_val
				

	def get_file_contents(self):
		'''
		Return the confidences and correct mask that the current node has seen.
		'''
		if not (isfile(self.conf_file) and isfile(self.corr_file)):
			return None, None
			
		if getsize(self.conf_file) == 0 or getsize(self.corr_file) == 0:
			return None, None
			
		return np.genfromtxt(self.conf_file), np.genfromtxt(self.corr_file).astype(np.bool)
		
		
	def reset(self):
		'''
		Remove the confidence and correct mask files on disk.
		'''
		if isfile(self.conf_file):
			os.remove(self.conf_file)
		
		if isfile(self.corr_file):
			os.remove(self.corr_file)
