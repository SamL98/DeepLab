import numpy as np
from functools import reduce

def fg_mask_for(gt):
	return ((gt > 0) & (gt < 255))

def set_fg_in_larger_array(fg, fg_mask, shape):
	if len(fg_mask.shape) > 1:
		fg_mask = fg_mask.ravel()

	flattened_length = reduce(lambda x, y: x*y, shape)
	assert flattened_length == len(fg_mask), f'Flattened length: {flattened_length}, Len fg_mask: {len(fg_mask)}'

	arr = np.zeros((flattened_length), dtype=fg.dtype)
	arr[np.where(fg_mask)[0]] = fg
	arr = arr.reshape(shape)

	return arr

def sm_of_logits(logits):
	logits_max = logits.max(-1)
	if len(logits.shape) > 1:
		logits_max = logits_max[...,np.newaxis]
	
	logits -= logits_max
	exp_logits = np.exp(logits)
	
	exp_logits_sum = exp_logits.sum(-1)
	if len(logits.shape) > 1:
		exp_logits_sum = exp_logits_sum[...,np.newaxis]
		
	sm = exp_logits / np.maximum(1e-7, exp_logits_sum)
	return sm
