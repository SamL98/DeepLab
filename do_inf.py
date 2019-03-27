import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys

from util import *

def perform_inference_on_image(idx, slices, args):
	'''
	Performs calibrated inference on a single image -- Get a confident mask at every slice and save to disk

	Params:
		idx: The index of the image within args.imset
		slices: A copy of the original slices
		args: The command line arguments
	'''
	logits, term_preds, gt_info = load_logit_pred_gt_triplet(args.imset, idx, ret_shape=True, ret_mask=True)
	gt, orig_shape, fg_mask = gt_info

	confident_masks = []
	confidence_maps = []

	if not args.sm_by_slice: scores = sm_of_logits(logits, start_idx=1, zero_pad=True)
	else: scores = logits

	for slc in slices:
		slc_fg_conf_mask = np.zeros_like(term_preds)
		slc_fg_conf_map = np.zeros((len(logits)), dtype=logits.dtype)

		slc_pred_labs = remap_label_arr(term_preds, slc)
		slc_scores = remap_scores_arr(scores, slc)

		if args.sm_by_slice: slc_sm = sm_of_logits(slc_scores)
		else: slc_sm = slc_scores

		for pix_idx, (slc_pred_lab, slc_sm_vec) in enumerate(zip(slc_pred_labs, slc_sm)):
			node = slc[slc_pred_lab]

			slc_sm_val = slc_sm_vec[slc_pred_lab]
			conf = node.get_conf_for_score(slc_sm_val)

			pred_lab = node.node_idx
			if len(node.terminals) == 1:
				pred_lab = node.terminals[0]

			slc_fg_conf_mask[pix_idx] = pred_lab
			slc_fg_conf_map[pix_idx] = conf

		slc_conf_mask = set_fg_in_larger_array(slc_fg_conf_mask, fg_mask, orig_shape)
		confident_masks.append(slc_conf_mask)

		slc_conf_map = set_fg_in_larger_array(slc_fg_conf_map, fg_mask, orig_shape)
		confidence_maps.append(slc_conf_map)

	save_calib_pred(args.imset, idx, args.name, confident_masks, confidence_maps)
	
def perform_inference_on_idxs(idxs, slices, args):
	'''
	Performs calibrated inference on the specified indices

	Params:
		idx: The indices into args.imset
		slices: A copy of the original slices
		args: The command line arguments
	'''
	for idx in idxs:
		perform_inference_on_image(idx, slices, args)
	
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
parser.add_argument('--sm_by_slice', dest='sm_by_slice', action='store_true', help='Whether or not to take the softmax of the logits at each slice of the hierarchy. True by default.')
parser.add_argument('--name', dest='name', type=str, help='The name of the current method.')

if __name__ == '__main__':
	args = parser.parse_args()

	slices = read_slices(args.slice_file, reset=False)
	args.nb = len(slices[0][0].acc_hist)
	param_batches = get_param_batches(slices, args)

	with poolcontext(args.num_proc) as p:
		_ = p.map(perform_inference_on_idxs_unpack, param_batches)
