import numpy as np

class Slice(object):
	def __init__(self, nodes, prev_slc_len):
		self.nodes = nodes

		lut = []
		for i in range(prev_slc_len):
			for j, node in enumerate(nodes):
				if i in node.children:
					lut.append(j)
					break

		self.label_lut = np.array(lut, dtype=np.uint8)

	def remap_scores_and_labels(self, scores, gts, term_preds):
		gts[:] = self.label_lut[gts]
		term_preds[:] = self.label_lut[term_preds]

		scores_out = np.zeros((len(scores), len(self.nodes)), dtype=scores.dtype)
		for i, node in enumerate(self.nodes):
			scores_out[:,i] = scores[:,node.children].sum(1)

		return scores_out

	def add_attr_if_not_exists(self, attr_name, attr_val):
		if not hasattr(self, attr_name):
			setattr(self, attr_name, attr_val)

	def remove_attr_if_exists(self, attr_name):
		if hasattr(self, attr_name):
			delattr(self, attr_name)

	def accum_node(self, node, slc_c_count=None, slc_ic_count=None):
		for attr in ['c_hist', 'ic_hist']:
			if not hasattr(node, attr):
				print(f'Node does not have {attr} attribute')
				return

		self.add_attr_if_not_exists('c_hist', np.zeros_like(node.c_hist))	
		self.add_attr_if_not_exists('ic_hist', np.zeros_like(node.ic_hist))	

		c_weight = 1
		if not slc_c_count is None and slc_c_count > 0:
			c_weight = node.c_hist.sum() / float(slc_c_count)

		ic_weight = 1
		if not slc_ic_count is None and slc_ic_count > 0:
			ic_weight = node.ic_hist.sum() / float(slc_ic_count)

		self.c_hist += node.c_hist * c_weight / float(len(self.nodes))
		self.ic_hist += node.ic_hist * ic_weight / float(len(self.nodes))

	def generate_acc_hist(self, alpha):
		self.tot_hist = self.c_hist + self.ic_hist
		self.tot_hist[self.tot_hist == 0] = 1

		acc_hist = self.c_hist.astype(np.float64) / np.maximum(1e-7, self.tot_hist.astype(np.float64))
		acc_hist = np.minimum(1, acc_hist)
		acc_hist, int_ranges = conf_ints(acc_hist, self.tot_hist, alpha)

		self.acc_hist = acc_hist
		self.int_ranges = int_ranges

	def get_acc_hist(self, lb=True):
		if lb: return self.acc_hist - self.int_ranges
		return self.acc_hist
			
	def conf_for_scores(self, scores, lb=True):
		if not hasattr(self, 'node_data'):
			assert isfile(join(self.node_data_fname))
			self.load_node_data()

		acc_hist = self.get_acc_hist(lb)
		return np.interp(scores, np.linspace(0, 1, num=len(acc_hist)), acc_hist)

	def reset(self, nb):
		self.remove_attr_if_exists('c_hist')
		self.remove_attr_if_exists('ic_hist')

	def __len__(self):
		return len(self.nodes)

	def __getitem__(self, index):
		return self.nodes[index]

	def __setitem__(self, index, value):
		self.nodes[index] = value

	def __delitem__(self, index):
		del self.nodes[index]

	def __iter__(self):
		self.node_idx = 0
		return self

	def __next__(self):
		if self.node_idx >= len(self.nodes):
			raise StopIteration
		else:
			self.node_idx += 1
			return self.nodes[self.node_idx-1]
