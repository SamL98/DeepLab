import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys

tree_fname = sys.argv[1]
slices = loadmat(tree_fname)['slices']

imset = 'val'
if len(sys.argv) > 2:
	imset = sys.argv[2]

ds_path = 'D:/datasets/processed/voc2012'
ds_info = loadmat(join(ds_path, 'dataset_info.mat'))
m = ds_info['num_'+imset]
nc = ds_info['num_classes']-1

logit_path = join(ds_path, imset, imset+'_%06d_logits.mat')
gt_path = join(ds_path, imset, imset+'_%06d_pixeltruth.mat')	

res = 0.05
nb = int(1./res)

corr_accum = [np.zeros((len(slc), nb), dtype=np.uint64) for slc in slices]
total_accum = [np.zeros((len(slc), nb), dtype=np.uint64) for slc in slices]

def confidence_for_cluster(sm_vec, cluster_idxes):
	return sm_vec[cluster_idxes].sum()

def remap_gt(true_label, slc):
	for i, cluster in enumerate(slc):
		if true_label in cluster: return i

vconf = np.vectorize(confidence_for_cluster)
vremap = np.vectorize(remap_gt)

for idx in range(1, m+1):
	logits = loadmat(logit_path % idx)['logits_img'][...,1:].reshape(-1, nc)
	exp_logits = np.exp(logits)
	sm = exp_logits / np.maximum(np.sum(exp_logits, axis=-1)[...,np.newaxis], 1e-7)
	gt = loadmat(gt_path % idx)['truth_img'].ravel()-1

	# discard pixels that were either background or void in ground truth 
	sm = sm[gt>=0 & gt<nc]
	gt = gt[gt>=0 & gt<nc]

	for i, slc in enumerate(slices):
		agg_sm = np.array([vconf(sm, cluster) for cluster in slc])
		agg_pred = np.argmax(agg_sm, axis=-1)
		agg_gt = vremap(gt, slc)

		for j in range(len(slc)):
			mask = agg_gt==j

			sm_masked = agg_sm[mask]
			pred_masked = agg_pred[mask]

			bins = np.floor(sm_masked/res).astype(np.uint8)
			corr_mask = pred_masked==j
			corr_accum[i][j,bins] += corr_mask
			total_accum[i][j,bins] += 1


recall_hists = [corr.astype(np.float32)/np.maximum(total, 1e-7) for corr, total in zip(corr_accum, total_accum)]
savemat('hists.mat', {'slice_histograms': recall_hists})
