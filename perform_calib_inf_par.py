import numpy as np
import multiprocessing as mp
from contextlib import contextmanager
from hdf5storage import loadmat, savemat
from os.path import join
import sys

from util import *

@contextmanager
def poolcontext(num_proc):
    pool = mp.Pool(num_proc)
    yield pool
    pool.terminate()

def calibrate_logits(idx, imset, slices, nb, save, conf_thresh=0.75, sm_by_slice=True):
	res = 1./nb
	
	logits = load_logits(imset, idx, reshape=False)
	logits = logits.reshape(-1, 21)

	gt = load_gt(imset, idx, reshape=False)
	predicted_mask = np.zeros(gt.shape, dtype=np.uint8)
	gt = gt.ravel()

	if not sm_by_slice:
		scores = sm_of_logits(logits, start_idx=1, zero_pad=True)
	else:
		scores = logits

	# Loop over each pixel's logit vector and its corresponding ground truth
	for pix_idx, (true_label, score_vec) in enumerate(zip(gt, scores)):
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

			# Get the predicted label (index within the cluster)
			pred_label = np.argmax(slc_sm)

			# Bin the softmax confidence for that label
			conf = slc_sm[pred_label]
			binno = np.floor(conf/res).astype(np.uint8)
			binno = min(binno, nb-1)

			# Get the calibrated confidence from the cluster's accuracy histogram
			acc_hist = slc[pred_label].acc_hist
			calib_conf = acc_hist[binno]

			if calib_conf >= conf_thresh:
				# If we predicted a terminal label in not the first slice, output the terminal label, not the node index
				if len(slc[pred_label].terminals) == 1:
					confident_label = slc[pred_label].terminals[0]
				else:
					confident_label = slc[pred_label].node_idx

				break

		r, c = np.unravel_index(pix_idx, predicted_mask.shape)
		predicted_mask[r, c] = confident_label
	
	if save:
		save_calib_pred(imset, idx, predicted_mask, conf_thresh)
	return predicted_mask
	
	
def calibrate_logits_unpack(params):
	calibrate_logits(*params)
	
from argparse import ArgumentParser
parser = ArgumentParser(description='Build the calibration hierarchy using multiprocessing.')
parser.add_argument('--slice_file', dest='slice_file', type=str, default='slices.pkl', help='The pickle file that specifies the hierarchy.')
parser.add_argument('--imset', dest='imset', type=str, default='val', help='The image set to build the calibration histograms from. Either val or test')
parser.add_argument('--num_proc', dest='num_proc', type=int, default=1, help='The number of processes to spawn to parallelize calibration.')

if __name__ == '__main__':
	args = parser.parse_args()

	# Load the slices from the specified file

	slices = read_slices(args.slice_file, reset=args.reset)
	nb = len(slices[0][0].count_hist)

	# Load the index ordering -- indexes are ordered by number of foreground pixels in descending order
	#
	# This way, if multiprocessing is used, all processes will be given approximately the same workload

	idx_ordering = None
	idx_ordering_fname = args.imset.lower() + '_ordered.txt'

	if not isfile(idx_ordering_fname):
		from order_by_num_fg import order_imset_by_num_fg
		idx_ordering = order_imset_by_num_fg(args.imset, save=True)
	else:
		with open(idx_ordering_fname) as f:
			idx_ordering = [int(idx) for idx in f.read().split('\n')]
	

	# Split the indexes up between processed to try and spread the work evenly

	idx_ordering = np.array(idx_ordering)
	param_batches = []

	for procno in range(args.num_proc):
		idx_batch = idx_ordering[procno::args.num_proc]
		param_batches.append((idx_batch, args.imset, slices.copy(), nb, True))

	with poolcontext(args.num_proc) as p:
		_ = p.map(calibrate_logits_unpack, param_batches)
