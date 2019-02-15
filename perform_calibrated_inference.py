import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys

from util import *

def calibrate_logits(idx, imset, slices, nb, conf_thresh=0.75, ret_conf=False, ds_factor=1):
	res = 1./nb
	
	ds_path = 'D:/datasets/processed/voc2012'
	logit_path = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_logits.mat')
	gt_path = join(ds_path, 'truth', imset, imset+'_%06d_pixeltruth.mat')
	
	logits = loadmat(logit_path % idx)['logits_img']
	
	if ds_factor > 1:
		assert(type(ds_factor) == int)
		logits = logits[::ds_factor, ::ds_factor].reshape(-1, 21)
	else:
		logits = logits.reshape(-1, 21)

	gt = loadmat(gt_path % idx)['truth_img']
	predicted_mask = np.zeros(gt.shape, dtype=np.uint8)
	if ret_conf:
		conf_mask = np.zeros(gt.shape, dtype=np.float64)
	gt = gt.ravel()

	# Loop over each pixel's logit vector and its corresponding ground truth
	for pix_idx, (true_label, logit_vec) in enumerate(zip(gt, logits)):
		# If the ground truth is background or void, ignore
		if true_label == 0 or true_label == 255: continue

		confident_label = 0
		if ret_conf:
			confident_conf = 0

		# Loop over each slice, breaking when the confidence threshold is hit
		for i, slc in enumerate(slices):
			# Remap the logits to the slice clusters
			slc_logits = remap_scores(logit_vec, slc)

			# Take the softmax of the remapped logits
			slc_exp_logits = np.exp(slc_logits)
			slc_sm = slc_exp_logits / np.maximum(np.sum(slc_exp_logits), 1e-7)

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
				confident_label = slc[pred_label].cluster_idx
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

	ds_path = 'D:/datasets/processed/voc2012'
	ds_info = loadmat(join(ds_path, 'dataset_info.mat'))
	#m = ds_info['num_'+imset]
	m = 1449-350
	nc = 20

	pred_path = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_calib_pred.mat')
	nb = len(slices[0][0].acc_hist)

	for idx in range(1, m+1):
		print('Performing inference on logit no. %d' % idx)
		predicted_mask = calibrate_logits(idx, imset, slices, nb)
		savemat(pred_path % idx, {'pred_img': predicted_mask})
