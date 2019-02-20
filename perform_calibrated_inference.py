import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys

from util import *

def calibrate_logits(idx, imset, slices, nb, conf_thresh=0.75, ret_conf=False, ds_factor=1, sm_by_slice=True):
	nb = len(slices[0][0].count_hist)
	res = 1./nb
	logits = load_logits(imset, idx, reshape=False)
	
	if ds_factor > 1:
		assert(type(ds_factor) == int)
		logits = logits[::ds_factor, ::ds_factor].reshape(-1, 21)
	else:
		logits = logits.reshape(-1, 21)


	gt = load_gt(imset, idx, reshape=False)
	predicted_mask = np.zeros(gt.shape, dtype=np.uint8)

	if ret_conf:
		conf_mask = np.zeros(gt.shape, dtype=np.float64)

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
		if ret_conf:
			confident_conf = 0

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

				if ret_conf:
					confident_conf = calib_conf
				break

		r, c = np.unravel_index(pix_idx, predicted_mask.shape)
		predicted_mask[r, c] = confident_label
		if ret_conf:
			conf_mask[r,c] = confident_conf
			
	if ds_factor > 1:
		from skimage.transform import resize
		predicted_mask = resize(predicted_mask, 
							(predicted_mask.shape[0]*ds_factor, predicted_mask.shape[1]*ds_factor), 
							order=0,
							preserve_range=True).astype(np.uint8)
		
	if ret_conf:
		return predicted_mask, conf_mask
	return predicted_mask
	

if __name__ == '__main__':
	tree_fname = sys.argv[1]
	slices = read_slices(tree_fname)

	imset = 'test'
	if len(sys.argv) > 2:
		imset = sys.argv[2]

	for idx in range(1, m+1):
		print('Performing inference on logit no. %d' % idx)
		sys.stdout.flush()

		predicted_mask = calibrate_logits(idx, imset, slices, nb)
		savemat(pred_path % idx, {'pred_img': predicted_mask})
