import os
from os.path import join, isfile, getsize
import numpy as np
from scipy.stats import norm

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
			bin_edge_fname = join(self.data_dir, '%s_bin_edges.txt' % self.uid)
			if isfile(bin_edge_fname):
				self.bin_file = bin_edge_fname
				self.bin_edges = np.genfromtxt(bin_edge_fname)

			acc_hist_fname = join(self.data_dir, '%s_acc_hist.txt' % self.uid)
			if isfile(acc_hist_fname):
				self.acc_file = acc_hist_fname
				self.acc_hist = np.genfromtxt(acc_hist_fname)

			int_range_fname = join(self.data_dir, '%s_conf_int_range.txt' % self.uid)
			if isfile(int_range_fname):
				self.int_file = int_range_fname
				self.int_ranges = np.genfromtxt(int_file)

				
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

			
	def generate_acc_hist(self, nb, slc_len, equa=True, lb=True, interp=True, alpha=0.75):
		'''
		Generates the accuracy histogram for the current node.

		:param nb: the number of bins in the histogram
		:param equa: whether or not to equalize the histogram
		:param lb: whether or not to store the lower bound of the accuracy in each bin (according to the Wilson confidence interval)
		:param interp: whether or not to interpolate the resulting histogram (to fix bins with no pixels)
		:param alpha: the confidence for the Wilson confidence interval -- only matters when lb=True
		'''
		assert isfile(self.conf_file), '%s conf file does not exist' % self.uid
		assert isfile(self.corr_file), '%s corr file does not exist' % self.uid
		
		if getsize(self.conf_file) == 0:
			return

		confs, correct_mask = self.get_file_contents()
		
		if len(confs) == 0:
			return
		
		if equa:
			# If we are equalizing, first create a histogram of the confidences with 10 times
			# the target number of bins to create the CDF from.
			#
			# This is because each node has such a large number of pixels, the CDF is already fairly equalized.
			conf_hist = np.histogram(confs, bins=nb*10)[0]
			cdf = np.cumsum(conf_hist)
			cdf = cdf / np.maximum(1e-7, cdf[-1])

			cdf_intervals = np.linspace(0, 1, num=nb+1)
			xp = np.linspace(0, 1, num=len(conf_hist))
			bin_edges = np.interp(cdf_intervals, cdf, xp)
		else:
			abs_lb = 1./slc_len
			bin_edges = np.linspace(abs_lb, 1, num=nb+1)

		bin_edges = np.array(bin_edges)
		self.bin_edges = bin_edges[:]

		self.bin_file = join(self.data_dir, '%s_bin_edges.txt' % self.uid)
		np.savetxt(self.bin_file, self.bin_edges)

		corr_hist = np.zeros((nb), dtype=np.float32)
		count_hist = np.zeros_like(corr_hist)

		for i, bin_edge in enumerate(self.bin_edges[1:]):
			bin_mask = (self.bin_edges[i] < confs) & (confs <= bin_edge)
			
			num_pix = bin_mask.sum()
			num_corr = correct_mask[bin_mask].sum()

			corr_hist[i] += num_corr
			count_hist[i] += num_pix

		self.acc_hist = corr_hist.astype(np.float32) / np.maximum(1e-7, count_hist.astype(np.float32))

		if lb:
			self.int_file = join(self.data_dir, '%s_conf_int_range.txt' % self.uid)
			int_ranges = []

			for i, (acc_val, bin_count) in enumerate(zip(self.acc_hist, count_hist)):
				acc_adj, conf_range = calculate_conf_interval(acc_val, bin_count, alpha)
				int_ranges.append(conf_range)
				self.acc_hist[i] = acc_adj

			self.int_ranges = np.array(int_ranges)
			np.savetxt(self.int_file, self.int_ranges)

		if interp:
			self.interp_acc_hist(count_hist)

		self.acc_file = join(self.data_dir, '%s_acc_hist.txt' % self.uid)
		np.savetxt(self.acc_file, self.acc_hist)


	def get_acc_lower_bound(self, alpha=0.75, interp=True):
		'''
		Modifies the accuracy histogram to contain the lower bound according to the Wilson confidence
		interval at each bin. This shouldn't be called if generate_acc_hist was called with lb=True.

		:param alpha: the confidence
		:param interp: whether or not to interpolate the resulting histogram
		'''
		assert alpha <= 1 and alpha >= 0
		assert isfile(self.conf_file), '%s conf file does not exist' % self.uid
		
		if getsize(self.conf_file) == 0:
			return

		confs = np.genfromtxt(self.conf_file)

		if not hasattr(self, 'acc_hist'):
			assert isfile(self.acc_file), '%s acc file does not exist' % self.uid
			self.acc_hist = np.genfromtxt(self.acc_file)

		if not hasattr(self, 'bin_edges'):
			assert isfile(self.bin_file), '%s bin file does not exist' % self.uid
			self.bin_edges = np.genfromtxt(self.bin_file)
			
		if interp:
			# We only need the corresponding count hist if we are interpolating afterwards
			count_hist = []

		for i, bin_edge in enumerate(self.bin_edges[1:]):
			bin_mask = (self.bin_edges[i] < confs) & (confs <= bin_edge)
			n = bin_mask.sum()

			if interp:
				count_hist.append(n)

			if n == 0:
				continue

			self.acc_hist[i] = calculate_conf_lower_bound(self.acc_hist[i], n, alpha)

		if interp:
			self.acc_hist = self.interp_acc_hist(count_hist)
			

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
		assert isfile(self.bin_file) and isfile(self.acc_file)
		
		if getsize(self.corr_file) == 0:
			return 0
		
		if not hasattr(self, 'bin_edges'):
			self.bin_edges = np.genfromtxt(self.bin_file)
			
		if not hasattr(self, 'acc_hist'):
			self.acc_hist = np.genfromtxt(self.acc_file)

		if not hasattr(self, 'int_ranges'):
			self.int_ranges = np.genfromtxt(self.int_file)
			
		for i, bin_edge in enumerate(self.bin_edges[1:]):
			if score <= bin_edge:
				# Return the first histogram value that is within the current bin
				acc_val = self.acc_hist[i]

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
