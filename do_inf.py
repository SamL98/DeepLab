import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys

from util import *

def calibrate_logits(idx, imset, slices, nb, save, conf_thresh, sm_by_slice, name):
	logits = load_logits(imset, idx, reshape=True)
	gt = load_gt(imset, idx, reshape=False)
	orig_shape = gt.shape

	gt = gt.ravel()
	tot_pred_mask = np.zeros_like(gt)

	fg_mask = fg_mask_for(gt)
	logits = logits[fg_mask]
	gt = gt[fg_mask]

	fg_pred_mask = np.zeros_like(gt)

	# Get the DeepLab terminal predictions, ignoring background
	term_preds = np.argmax(logits[:,1:], -1) + 1

	if not sm_by_slice: scores = sm_of_logits(logits, start_idx=1, zero_pad=True)
	else: scores = logits

	for pix_idx, (term_pred, score_vec, gt_lab) in enumerate(zip(term_preds, scores, gt)):
		for i, slc in enumerate(slices):
			slc_score = remap_scores(score_vec, slc)

			if sm_by_slice: slc_sm = sm_of_logits(slc_score)
			else: slc_sm = slc_score

			slc_pred_lab = remap_label(term_pred, slc)
			slc_term_pred_lab = remap_label(term_pred, slc, push_down=True)

			node = slc[slc_pred_lab]
			conf = node.get_conf_for_score(slc_sm[slc_pred_lab])

			if conf >= conf_thresh:
				fg_pred_mask[pix_idx] = slc_term_pred_lab
				break
		
	tot_pred_mask[fg_mask] = fg_pred_mask
	tot_pred_mask = tot_pred_mask.reshape(orig_shape)
	if save: 
		save_calib_pred(imset, idx, tot_pred_mask, conf_thresh, name)
	return pred_mask
	
	
def calibrate_logits_unpack(params):
	idx_batch, slices, args = params
	for idx in idx_batch:
		calibrate_logits(idx, args.imset, slices, args.nb, args.save, args.conf_thresh, args.sm_by_slice, args.name)
	
from argparse import ArgumentParser
parser = ArgumentParser(description='Build the calibration hierarchy using multiprocessing.')
parser.add_argument('--slice_file', dest='slice_file', type=str, default='slices.pkl', help='The pickle file that specifies the hierarchy.')
parser.add_argument('--imset', dest='imset', type=str, default='test', help='The image set to build the calibration histograms from. Either val or test')
parser.add_argument('--num_proc', dest='num_proc', type=int, default=8, help='The number of processes to spawn to parallelize calibration.')
parser.add_argument('--save', dest='save', action='store_false', help='Whether or not to save the inference results.')
parser.add_argument('--conf_thresh', dest='conf_thresh', type=float, default=0.75, help='The confidence threshold for inference.')
parser.add_argument('--sm_by_slice', dest='sm_by_slice', action='store_true', help='Whether or not to take the softmax of the logits at each slice of the hierarchy. True by default.')
parser.add_argument('--name', dest='name', type=str, help='The name of the current method.')

if __name__ == '__main__':
	args = parser.parse_args()

	slices = read_slices(args.slice_file, reset=False)
	args.nb = len(slices[0][0].acc_hist)
	param_batches = get_param_batches(slices, args)

	with poolcontext(args.num_proc) as p:
		_ = p.map(calibrate_logits_unpack, param_batches)
