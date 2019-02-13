import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys

from util import *

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

logit_path = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_logits.mat')
gt_path = join(ds_path, 'truth', imset, imset+'_%06d_pixeltruth.mat')
pred_path = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_calib_pred.mat')

nb = len(slices[0][0].acc_hist)
res = 1./nb

conf_thresh = 0.75

for idx in range(1, m+1):
	print('Performing inference on logit no. %d' % idx)

	logits = loadmat(logit_path % idx)['logits_img'].reshape(-1, nc+1)
	logits[:,0] = 0

	gt = loadmat(gt_path % idx)['truth_img']
	predicted_mask = np.zeros(gt.shape, dtype=np.uint8)
	gt = gt.ravel()

	# Loop over each pixel's logit vector and its corresponding ground truth
	for pix_idx, (true_label, logit_vec) in enumerate(zip(gt, logits)):
		# If the ground truth is background or void, ignore
		if true_label == 0 or true_label == 255: continue

		confident_label = 0

		# Loop over each slice, breaking when the confidence threshold is hit
		for i, slc in enumerate(slices):
			# Remap the logits to the slice clusters
			slc_logits = remap_logits(logit_vec, slc)

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
				break

		r, c = np.unravel_index(pix_idx, predicted_mask.shape)
		predicted_mask[r, c] = confident_label

	savemat(pred_path % idx, {'pred_img': predicted_mask})
