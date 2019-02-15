import numpy as np
from hdf5storage import loadmat
from os.path import join
	
import sys
imset = sys.argv[1].lower().capitalize()

DS_PATH = 'D:/datasets/processed/voc2012' 
PRED_PATH = join(DS_PATH, 'Deeplab_Prediction', imset)
GT_PATH = join(DS_PATH, 'Truth', imset)

ds_info = loadmat(join(DS_PATH, 'dataset_info.mat'))
#num_img = ds_info['num_'+imset.lower()]
num_img = 350
if imset.lower() == 'test':
	num_img = 1449-num_img

LGT_FMT = join(PRED_PATH, imset.lower()+'_%06d_logits.mat')
LGT_MAT_NAME = 'logits_img'

GT_FMT = join(GT_PATH, imset.lower()+'_%06d_pixeltruth.mat')
GT_MAT_NAME = 'truth_img'
	
acc_res = 0.05
correct_conf_hist = np.zeros((int(1./acc_res)), dtype=np.uint64)
incorrect_conf_hist = np.zeros_like(correct_conf_hist)
	
for idx in range(1, num_img+1):
	gt = loadmat(GT_FMT % idx)[GT_MAT_NAME].ravel()
	mask = (gt>0) & (gt<255)

	logits = loadmat(LGT_FMT % idx)[LGT_MAT_NAME].reshape(-1, 21)
	logits[:,0] = 0
	
	gt = gt[mask]
	logits = logits[mask]
	
	pred = np.argmax(logits, axis=-1)
	
	exp_logits = np.exp(logits)
	sm = exp_logits / np.maximum(1e-7, exp_logits.sum(-1)[:,np.newaxis])
	scores = np.max(sm, axis=-1)
	
	bins = np.floor(scores/acc_res).astype(np.uint8)
	bins = np.minimum(bins, correct_conf_hist.shape[0]-1)
	
	corr_bins = bins[pred==gt]
	incorr_bins = bins[pred!=gt]
	
	for binno in np.unique(bins):
		correct_conf_hist[binno] += np.sum(corr_bins==binno)
		incorrect_conf_hist[binno] += np.sum(incorr_bins==binno)

		
hist_len = correct_conf_hist.shape[0]
correct_conf_hist = correct_conf_hist/float(correct_conf_hist.sum())
incorrect_conf_hist = incorrect_conf_hist/float(incorrect_conf_hist.sum())
		
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(range(hist_len), correct_conf_hist, align='edge')
ax.set_xlabel('Confidence')
ax.xaxis.set_ticks(np.linspace(0, hist_len, num=6))
ax.set_xticklabels(['%.1f' % v for v in np.linspace(0.0, 1.0, num=6)])
ax.set_ylabel('% of correct pixels')
ax.yaxis.set_ticks(np.linspace(0.0, 1.0, num=6))
ax.set_title('Confidence on %s Set When Correct' % imset)

fig.savefig('images/conf_corr_hist.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
ax.bar(range(hist_len), incorrect_conf_hist, align='edge')
ax.set_xlabel('Confidence')
ax.xaxis.set_ticks(np.linspace(0, hist_len, num=6))
ax.set_xticklabels(['%.1f' % v for v in np.linspace(0.0, 1.0, num=6)])
ax.set_ylabel('% of incorrect pixels')
ax.yaxis.set_ticks(np.linspace(0.0, 1.0, num=6))
ax.set_title('Confidence on %s Set When Incorrect' % imset)

fig.savefig('images/conf_incorr_hist.png', bbox_inches='tight')
plt.show()