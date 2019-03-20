import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys

from util import *

# For each slice:
#
# Generate the confident mask and predictions for that slice
# 
# On first slice, conf_mask will have same length as pred_mask
#
# Then mask logits, gt, and pred_mask by unconf_mask

def generate_calibrated_mask(res, nb, slc_idx, slices, scores, gt, term_preds, conf_thresh, sm_by_slice):
	

def calibrate_logits(idx, imset, slices, nb, save, conf_thresh, sm_by_slice, name):
	logits = load_logits(imset, idx, reshape=True)
	gt = load_gt(imset, idx, reshape=False)
	mask_shape = gt.shape

	gt = gt.ravel()
	pred_mask = np.zeros_like(gt)

	fg_mask = fg_mask_for(gt)
	logits = logits[fg_mask]
	gt = gt[fg_mask]

	term_preds = np.argmax(logits[:,1:], -1) + 1

	tot_mask = fg_mask

	if not sm_by_slice: scores = sm_of_logits(logits, start_idx=1, zero_pad=True)
	else: scores = logits

	for slc in slices:
		# Create a table for calibration where each row is the accuracy histogram for that node
		calib_table = []
		for node in slc:
			calib_table.append(node.get_conf_acc_hist())
		calib_table = np.array(calib_table)

		# Remap the logits or softmax and terminal predictions to the current slice
		slc_scores = np.zeros((len(scores), len(slc)), dtype=np.float32)
		slc_pred_labels = np.zeros((len(scores)), dtype=np.uint8)
		slc_term_pred_labels = np.zeros((len(scores)), dtype=np.uint8)

		for i, (score_vec, term_pred) in enumerate(zip(scores, term_preds)):
			slc_scores[i] = remap_scores(score_vec, slc)
			slc_pred_labels[i] = remap_label(term_pred, slc)
			slc_term_pred_labels[i] = remap_label(term_pred, slc, push_down=True)

		if sm_by_slice: slc_sm = sm_of_logits(slc_scores)
		else: slc_sm = slc_scores

		# Obtain the calibrated confidence for every logit on the forced path
		# Not buggy, only buggy before with binvec because of increment
		binvec = np.floor(slc_sm[:,slc_pred_labels]/res).astype(np.int16)
		binvec = np.minimum(binvec, nb-1)
		confs = calib_table[slc_pred_labels, binvec]

		conf_mask = confs >= conf_thresh
		tot_mask = np.where(tot_mask)[0][conf_mask]
		pred_mask[tot_mask] = slc_term_pred_labels[conf_mask]

		unconf_mask = (not conf_mask)
		if unconf_mask.sum() == 0:
			return predicted_mask

		scores = scores[unconf_mask]
		gt = gt[unconf_mask]
		term_preds = term_preds[unconf_mask]
		
	pred_mask = pred_mask.reshape(mask_shape)
	if save: 
		save_calib_pred(imset, idx, pred_mask, conf_thresh, name)
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
