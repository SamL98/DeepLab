import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys

from util import *

def calibrate_logits(idx, imset, slices, nb, save, conf_thresh, sm_by_slice, name):
	res = 1./nb
	
	logits = load_logits(imset, idx, reshape=True)

	gt = load_gt(imset, idx, reshape=False)
	predicted_mask = np.zeros(gt.shape, dtype=np.uint8)
	gt = gt.ravel()

	if not sm_by_slice:
		scores = sm_of_logits(logits, start_idx=1, zero_pad=True)
	else:
		scores = logits

	fg_mask = fg_mask_for(gt)
	gt = gt[fg_mask]

	scores = scores[fg_mask]
	terminal_preds = np.argmax(scores, axis=-1)

	for slc in slices:
		slc_scores = np.zeros((len(logits), len(slc)), dtype=np.float32)
		pred_labels = np.zeros((len(logits)), dtype=np.uint8)

		for i, (score_vec, term_pred) in enumerate(zip(scores, terminal_preds)):
			slc_scores[i] = remap_scores(score_vec, slc)
			pred_labels[i] = remap_label(term_pred, slc)

		if sm_by_slice:
			slc_sm = sm_of_logits(slc_scores)
		else:
			slc_sm = slc_scores


	# Loop over each pixel's logit vector and its corresponding ground truth
	for pix_idx, (true_label, score_vec, ) in enumerate(zip(gt, scores, terminal_preds)):
		# If the ground truth is background or void, ignore
		if true_label == 0 or true_label == 255: continue

		confident_label = 0

		# Loop over each slice, breaking when the confidence threshold is hit
		for i, slc in enumerate(slices):
			if sm_by_slice:
				# Remap the logits to the slice clusters
				slc_logits = remap_scores(score_vec, slc)

				# Take the softmax of the remapped logits
				slc_sm = sm_of_logits(slc_logits)
			else:
				slc_sm = remap_scores(score_vec, slc)

			# Remap the terminal prediction to the slice
			pred_label = remap_label(terminal_preds[pix_idx], slc)

			node = slc[pred_label]
			calib_conf = node.get_conf_for_score(slc_sm[pred_label])

			if calib_conf >= conf_thresh:
				# If we predicted a terminal label in not the first slice, output the terminal label, not the node index
				if len(slc[pred_label].terminals) == 1:
					confident_label = node.terminals[0]
				else:
					confident_label = node.node_idx

				# Break if we are confident enough
				break

		r, c = np.unravel_index(pix_idx, predicted_mask.shape)
		predicted_mask[r, c] = confident_label
	
	if save:
		save_calib_pred(imset, idx, predicted_mask, conf_thresh, name)

	return predicted_mask
	
	
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
