import pickle
import numpy as np

def read_slices(fname, reset=False):
	with open(fname, 'rb') as f:
		slices = pickle.load(f)

	if reset:
		for slc in slices:
			for node in slc:
				node.__init__(node.name, node.node_idx, node.terminals, is_main=True)
	return slices

def save_slices(fname, slices):
	with open(fname, 'wb') as f:
		pickle.dump(slices, f)

def confidence_for_node(vec, node):
	return vec[node.terminals].sum()

def remap_label(term_label, slc):
	for i, node in enumerate(slc):
		if term_label in node.terminals:
			return i

def remap_label_arr(label_arr, slc):
	return np.array([remap_label(lab, slc) for lab in label_arr])
		
def remap_scores(vec, slc):
	conf = []
	for node in slc:
		conf.append(confidence_for_node(vec, node))
	return np.array(conf)

def remap_scores_arr(score_arr, slc):
	return np.array([remap_scores(score_vec, slc) for score_vec in score_arr])