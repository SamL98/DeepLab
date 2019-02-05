import numpy as np
from hdf5storage import loadmat
from os.path import join
	
import sys
imset = sys.argv[1].lower().capitalize()

DS_PATH = 'D:/datasets/processed/voc2012' 
PRED_PATH = join(DS_PATH, 'Deeplab_Prediction', imset)
GT_PATH = join(DS_PATH, 'Truth', imset)

ds_info = loadmat(join(DS_PATH, 'dataset_info.mat'))
num_img = ds_info['num_'+imset.lower()]
nc = 21
class_names = ds_info['class_labels']

SM_FMT = '%s_%06d_softmax.mat'
SM_MAT_NAME = 'softmax_img'

PRED_FMT = '%s_%06d_prediction.mat'
PRED_MAT_NAME = 'pred_img'

GT_FMT = '%s_%06d_pixeltruth.mat'
GT_MAT_NAME = 'truth_img'

def load_sm_pred_gt(idx):
	global PRED_PATH, SM_FMT, SM_MAT_NAME, GT_PATH, GT_FMT, GT_MAT_NAME, imset
	sm = loadmat(join(PRED_PATH, SM_FMT % (imset, idx)))[SM_MAT_NAME].flatten()
	pred = loadmat(join(PRED_PATH, PRED_FMT % (imset, idx)))[PRED_MAT_NAME].flatten()
	gt = loadmat(join(GT_PATH, GT_FMT % (imset, idx)))[GT_MAT_NAME].flatten()
	return sm, pred, gt
	
	
acc_res = 0.1
hists = []
histlen = int(1./acc_res)

for _ in range(nc):
	correct_hist = np.zeros((histlen), dtype=np.uint64)
	count_hist = np.zeros_like(correct_hist)
	hists.append((correct_hist, count_hist))
	
for i in range(1, num_img+1):
	scores, pred, gt = load_sm_pred_gt(i)
	bins = np.floor(scores/acc_res).astype(np.uint8)
	
	for c in range(nc):
		if (gt==c).sum() == 0:
			continue
			
		gt_mask = gt[gt==c]
		pred_mask = pred[gt==c]
		bin_mask = bins[gt==c]
		
		for binno in np.unique(bin_mask):
			pred_bin = pred_mask[bin_mask==binno]
			gt_bin = gt_mask[bin_mask==binno]
			hists[c][0][min(binno, histlen-1)] += np.sum(pred_bin==gt_bin)
			hists[c][1][min(binno, histlen-1)] += np.sum(bin_mask==binno)
		

for i, (correct_hist, count_hist) in enumerate(hists):
	acc_hist = correct_hist.astype(np.float64)/np.maximum(1e-7, count_hist)
	class_name = class_names[i]
		
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	ax.bar(range(acc_hist.shape[0]), [v*acc_res for v in range(acc_hist.shape[0])], align='edge', color='r', alpha=0.5)
	ax.bar(range(acc_hist.shape[0]), acc_hist, align='edge')
	ax.set_xlabel('Confidence')
	ax.xaxis.set_ticks(np.linspace(0, acc_hist.shape[0], num=6))
	ax.set_xticklabels(['%.1f' % v for v in np.linspace(0.0, 1.0, num=6)])
	ax.set_ylabel('Accuracy')
	ax.yaxis.set_ticks(np.linspace(0.0, 1.0, num=6))
	ax.set_title('%s' % class_name)
	fig.savefig('acc_hists/%s_acc_hist.png' % class_name, bbox_inches='tight')
	plt.close(fig)