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
	
	
acc_res = 0.1
correct_hist = np.zeros((int(1./acc_res)), dtype=np.uint64)
count_hist = np.zeros_like(correct_hist)
	
for i in range(1, num_img+1):
	scores, pred, gt = load_sm_pred_gt(i)
	bins = np.floor(scores/acc_res).astype(np.uint8)
	
	for binno in np.unique(bins):
		pred_bin = pred[bins==binno]
		gt_bin = gt[bins==binno]
		correct_hist[min(binno, correct_hist.shape[0]-1)] += np.sum(pred_bin==gt_bin)
		count_hist[min(binno, count_hist.shape[0]-1)] += np.sum(bins==binno)
		

acc_hist = correct_hist.astype(np.float64)/np.maximum(1e-7, count_hist)
		
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(range(acc_hist.shape[0]), [v*acc_res for v in range(acc_hist.shape[0])], align='edge', color='r', alpha=0.5)
ax.bar(range(acc_hist.shape[0]), acc_hist, align='edge')
ax.set_xlabel('Confidence')
ax.xaxis.set_ticks(np.linspace(0, acc_hist.shape[0], num=6))
ax.set_xticklabels(['%.1f' % v for v in np.linspace(0.0, 1.0, num=6)])
ax.set_ylabel('Accuracy')
ax.yaxis.set_ticks(np.linspace(0.0, 1.0, num=6))
#ax.plot(np.linspace(0, acc_hist.shape[0], num=6), np.linspace(0.0, 1.0, num=6))
ax.set_title('%s Set' % imset)
fig.savefig('acc_hist.png', bbox_inches='tight')
plt.show()