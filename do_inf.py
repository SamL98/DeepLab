import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys
from functools import reduce

from util import *

def perform_inference_on_image(idx, slices, args, ret_mask=False):
	'''
	Performs calibrated inference on a single image

	Params:
		idx: The index of the image within args.imset
		slices: A copy of the original slices
		args: The command line arguments
		ret_mask: Whether or not to return the predicted mask from this function

	Returns:
		The predicted mask if ret_mask = True
	'''
	logits, term_preds, gt_info = load_logit_pred_gt_triplet(args.imset, idx, ret_shape=True, ret_mask=True)
	gt, orig_shape, fg_mask = gt_info

	fg_pred_mask = np.zeros_like(term_preds)

	if not args.sm_by_slice: scores = sm_of_logits(logits, start_idx=1, zero_pad=True)
	else: scores = logits

	for pix_idx, (term_pred, score_vec) in enumerate(zip(term_preds, scores)):
		for i, slc in enumerate(slices):
			slc_score = remap_scores(score_vec, slc)
			slc_pred_lab = remap_label(term_pred, slc)

			if args.sm_by_slice: slc_sm = sm_of_logits(slc_score)
			else: slc_sm = slc_score

			node = slc[slc_pred_lab]
			slc_sm_val = slc_sm[slc_pred_lab]
			conf = node.get_conf_for_score(slc_sm_val)

			if conf >= args.conf_thresh:
				pred_lab = node.node_idx
				if len(node.terminals) == 1:
					pred_lab = node.terminals[0]

				fg_pred_mask[pix_idx] = pred_lab
				break
		
	tot_num_pix = reduce(lambda x, y: x*y, orig_shape)
	tot_pred_mask = np.zeros((tot_num_pix), dtype=fg_pred_mask.dtype)

	tot_pred_mask[fg_mask] = fg_pred_mask

	tot_pred_mask = tot_pred_mask.reshape(orig_shape)

	if save: 
		save_calib_pred(args.imset, idx, tot_pred_mask, args.conf_thresh, args.name)

	if ret_mask:
		return tot_pred_mask
	
def perform_inference_on_idxs(idxs, slices, args):
	'''
	Performs calibrated inference on the specified indices

	Params:
		idx: The indices into args.imset
		slices: A copy of the original slices
		args: The command line arguments
	'''
	for idx in idxs:
		perform_inference_on_image(idx, slices, args, ret_mask=False)
	
def perform_inference_on_idxs_unpack(params):
	'''
	Wrapper for perform_inference_on_idxs
	'''
	perform_inference_on_idxs(*params)
	
from argparse import ArgumentParser
parser = ArgumentParser(description='Build the calibration hierarchy using multiprocessing.')
parser.add_argument('--slice_file', dest='slice_file', type=str, default='slices.pkl', help='The pickle file that specifies the hierarchy.')
parser.add_argument('--imset', dest='imset', type=str, default='test', help='The image set to build the calibration histograms from. Either val or test')
parser.add_argument('--num_proc', dest='num_proc', type=int, default=8, help='The number of processes to spawn to parallelize calibration.')
parser.add_argument('--dont_save', dest='save', action='store_false', help='Whether or not to save the inference results.')
parser.add_argument('--conf_thresh', dest='conf_thresh', type=float, default=0.75, help='The confidence threshold for inference.')
parser.add_argument('--sm_by_slice', dest='sm_by_slice', action='store_true', help='Whether or not to take the softmax of the logits at each slice of the hierarchy. True by default.')
parser.add_argument('--name', dest='name', type=str, help='The name of the current method.')

if __name__ == '__main__':
	args = parser.parse_args()

	slices = read_slices(args.slice_file, reset=False)
	args.nb = len(slices[0][0].acc_hist)
	param_batches = get_param_batches(slices, args)

	with poolcontext(args.num_proc) as p:
		_ = p.map(perform_inference_on_idxs_unpack, param_batches)
