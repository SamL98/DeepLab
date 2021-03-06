import numpy as np

def fg_mask_for(gt):
	return ((gt > 0) & (gt < 255))

def set_fg_in_larger_array(fg, fg_mask, shape):
	assert len(fg.shape) == 1
	flattened_length = reduce(lambda x, y: x*y, shape)
	assert flattened_length == len(fg)

	arr = np.zeros((flattened_length), dtype=fg.dtype)
	arr[fg_mask] = fg
	arr = arr.reshape(shape)

	return arr

def sm_of_logits(logits, start_idx=0, zero_pad=False):
	logits = logits[...,start_idx:]
	
	logits_max = logits.max(-1)
	if len(logits.shape) > 1:
		logits_max = logits_max[:,np.newaxis]
	
	logits -= logits_max
	exp_logits = np.exp(logits)
	
	exp_logits_sum = exp_logits.sum(-1)
	if len(logits.shape) > 1:
		exp_logits_sum = exp_logits_sum[:,np.newaxis]
		
	sm = exp_logits / np.maximum(1e-7, exp_logits_sum)
	
	if zero_pad:
		zero_vec = np.zeros((len(sm)), dtype=sm.dtype)[:,np.newaxis]
		sm = np.concatenate((zero_vec, sm), axis=1)

	return sm