import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys

from util import *

tree_fname = sys.argv[1]
slices = read_slices(tree_fname)

imset = 'val'
if len(sys.argv) > 2:
	imset = sys.argv[2]

ds_path = 'D:/datasets/processed/voc2012'
ds_info = loadmat(join(ds_path, 'dataset_info.mat'))
#m = ds_info['num_'+imset]
m = 350
nc = 20

logit_path = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_logits.mat')
gt_path = join(ds_path, 'truth', imset, imset+'_%06d_pixeltruth.mat')	

nb = len(slices[0][0].acc_hist)
res = 1./nb

for idx in range(1, m+1):
	print('Binning logit no. %d' % idx)

	logits = loadmat(logit_path % idx)['logits_img'][...,1:].reshape(-1, nc)
	zero_vec = np.zeros((len(logits)), dtype=logits.dtype)[...,np.newaxis] # shape = (len(logits), 1)
	logits = np.concatenate((zero_vec, logits), axis=1)

	gt = loadmat(gt_path % idx)['truth_img'].ravel()

	# discard pixels that were either background or void in ground truth 
	fg_mask = (gt>0) & (gt<=nc)
	logits = logits[fg_mask]
	gt = gt[fg_mask]

	for i, slc in enumerate(slices):
		slc_logits = np.array([remap_logits(logit_vec, slc) for logit_vec in logits])
		slc_exp_logits = np.exp(slc_logits)
		slc_sm = slc_exp_logits / np.maximum(np.sum(slc_exp_logits, axis=-1)[...,np.newaxis], 1e-7)

		for j, cluster in enumerate(slc):
			for true_label, sm_vec in zip(gt, slc_sm):
				label = remap_gt(true_label, slc)
				if label != j: continue
				
				conf = sm_vec[j]
				binno = np.floor(conf/res).astype(np.uint8)
				binno = min(binno, nb-1)

				if np.argmax(sm_vec) == label:
					cluster.corr_hist[binno] += 1

				cluster.count_hist[binno] += 1


for slc in slices:
	for cluster in slc:
		cluster.acc_hist = cluster.corr_hist.astype(np.float32) / cluster.count_hist.astype(np.float32)

save_slices(tree_fname, slices)
