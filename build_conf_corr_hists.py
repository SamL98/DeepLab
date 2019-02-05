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
	
	
acc_res = 0.05
correct_conf_hist = np.zeros((int(1./acc_res)), dtype=np.uint64)
incorrect_conf_hist = np.zeros_like(correct_conf_hist)
	
for i in range(1, num_img+1):
	scores, pred, gt = load_sm_pred_gt(i)
	bins = np.floor(scores/acc_res).astype(np.uint8)
	corr_bins = bins[pred==gt]
	incorr_bins = bins[pred!=gt]
	
	for binno in np.unique(bins):
		correct_conf_hist[min(binno, correct_conf_hist.shape[0]-1)] += np.sum(corr_bins==binno)
		incorrect_conf_hist[min(binno, incorrect_conf_hist.shape[0]-1)] += np.sum(incorr_bins==binno)

		
hist_len = correct_conf_hist.shape[0]
correct_conf_hist = correct_conf_hist/float(correct_conf_hist.sum())
incorrect_conf_hist = incorrect_conf_hist/float(incorrect_conf_hist.sum())
		
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(range(hist_len), correct_conf_hist, align='edge')
ax.set_xlabel('Confidence')
ax.xaxis.set_ticks(np.linspace(0, hist_len, num=6))
ax.set_xticklabels(['%.1f' % v for v in np.linspace(0.0, 1.0, num=6)])
ax.set_ylabel('Accuracy')
ax.yaxis.set_ticks(np.linspace(0.0, 1.0, num=6))
ax.set_title('%s Set Correct' % imset)

fig.savefig('conf_corr_hist.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
ax.bar(range(hist_len), incorrect_conf_hist, align='edge')
ax.set_xlabel('Confidence')
ax.xaxis.set_ticks(np.linspace(0, hist_len, num=6))
ax.set_xticklabels(['%.1f' % v for v in np.linspace(0.0, 1.0, num=6)])
ax.yaxis.set_ticks(np.linspace(0.0, 1.0, num=6))
ax.set_title('%s Set Incorrect' % imset)

fig.savefig('conf_incorr_hist.png', bbox_inches='tight')
plt.show()