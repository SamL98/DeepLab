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

	def remap_labels(self, labels):
		labels[...] = self.label_lut[labels]

	def remap_sm(self, sm):
		sm_out = np.zeros((sm.shape[0], sm.shape[1], len(self.nodes)), dtype=sm.dtype)
		for i, node in enumerate(self.nodes):
			sm_out[...,i] = scores[...,node.children].sum(-1)
		return sm_out 

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
