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

	def remap_scores_and_labels(scores, gts, term_preds):
		gts[:] = self.label_lut[gts]
		term_preds[:] = self.label_lut[term_preds]

		scores_out = np.zeros((len(scores), len(self.nodes)), dtype=scores.dtype)
		for i, node in enumerate(self.nodes):
			scores_out[:,i] = scores[:,node.children].sum(1)

		return scores_out

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
