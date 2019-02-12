import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys

from util import read_slices

tree_fname = sys.argv[1]
slices = loadmat(tree_fname)['slices']

imset = 'val'
if len(sys.argv) > 2:
	imset = sys.argv[2]

ds_path = 'D:/datasets/processed/voc2012'
ds_info = loadmat(join(ds_path, 'dataset_info.mat'))
m = ds_info['num_'+imset]
nc = 20

logit_path = join(ds_path, 'deeplab_prediction', imset, imset+'_%06d_logits.mat')
gt_path = join(ds_path, 'truth', imset, imset+'_%06d_pixeltruth.mat')	

res = 0.05
nb = int(1./res)

corr_accum = [np.zeros((len(slc), nb), dtype=np.uint64) for slc in slices]
total_accum = [np.zeros((len(slc), nb), dtype=np.uint64) for slc in slices]

def confidence_for_cluster(sm_vec, cluster_idxes):
	return sm_vec[cluster_idxes].sum()

def remap_gt(true_label, slc):
	for i, cluster in enumerate(slc):
		if true_label in cluster: return i
		
def remap_sm(sm_vec, slc):
	conf = []
	for cluster in slc:
		conf.append(confidence_for_cluster(sm_vec, cluster))
	return np.array(conf)

for idx in range(1, m+1):
	print('Binning logit no. %d' % idx)

	logits = loadmat(logit_path % idx)['logits_img'][...,1:].reshape(-1, nc)
	exp_logits = np.exp(logits)
	sm = exp_logits / np.maximum(np.sum(exp_logits, axis=-1)[...,np.newaxis], 1e-7)
	gt = loadmat(gt_path % idx)['truth_img'].ravel()-1

	# discard pixels that were either background or void in ground truth 
	sm = sm[(gt>=0) & (gt<nc)]
	gt = gt[(gt>=0) & (gt<nc)]

	for i, slc in enumerate(slices):
		for j, cluster in enumerate(slc):
			for true_label, sm_vec in zip(gt, sm):
				label = remap_gt(true_label, slc)
				if label != j: continue
				
				slc_sm = remap_sm(sm_vec, slc)
				
				conf = slc_sm[j]
				binno = np.floor(conf/res).astype(np.uint8)
				binno = min(binno, nb-1)

				if np.argmax(slc_sm) == label:
					corr_accum[i][j,binno] += 1
				total_accum[i][j,binno] += 1


recall_hists = [corr.astype(np.float32)/np.maximum(total, 1e-7) for corr, total in zip(corr_accum, total_accum)]
savemat('hists.mat', {'slice_histograms': recall_hists})
