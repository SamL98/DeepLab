import numpy as np
from hdf5storage import loadmat
from os.path import join
	
import sys
imset = sys.argv[1].lower().capitalize()

DS_PATH = 'D:/datasets/processed/voc2012' 
PRED_PATH = join(DS_PATH, 'Deeplab_Prediction', imset)
GT_PATH = join(DS_PATH, 'Truth', imset)

ds_info = loadmat(join(DS_PATH, 'dataset_info.mat'))
num_img = 350
if imset.lower == 'test':
	num_img = 1449-num_img

LGT_FMT = join(PRED_PATH, imset.lower()+'_%06d_logits.mat')
LGT_MAT_NAME = 'logits_img'

GT_FMT = join(GT_PATH, imset.lower()+'_%06d_pixeltruth.mat')
GT_MAT_NAME = 'truth_img'
	
acc_res = 0.1
correct_hist = np.zeros((int(1./acc_res)), dtype=np.uint64)
count_hist = np.zeros_like(correct_hist)
	
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
	
	for binno in np.unique(bins):
		pred_bin = pred[bins==binno]
		gt_bin = gt[bins==binno]
		correct_hist[min(binno, correct_hist.shape[0]-1)] += np.sum(pred_bin==gt_bin)
		count_hist[min(binno, count_hist.shape[0]-1)] += np.sum(bins==binno)
		

acc_hist = correct_hist.astype(np.float64)/np.maximum(1e-7, count_hist)

below_vals, above_vals = [], []
colors = []

for i, acc in enumerate(acc_hist):
	val = i*acc_res
	
	if acc < val:
		below_vals.append(acc)
		above_vals.append(val-acc)
		colors.append('r')
	else:
		below_vals.append(val)
		above_vals.append(acc-val)
		colors.append('tab:orange')
		
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

fig, ax = plt.subplots()

ax.bar(range(len(below_vals)), below_vals, align='edge')
ax.bar(range(len(above_vals)), above_vals, align='edge', bottom=below_vals, color=colors)

ax.set_xlabel('Confidence')
ax.xaxis.set_ticks(np.linspace(0, acc_hist.shape[0], num=6))
ax.set_xticklabels(['%.1f' % v for v in np.linspace(0.0, 1.0, num=6)])

ax.set_ylabel('Accuracy')
ax.yaxis.set_ticks(np.linspace(0.0, 1.0, num=6))

ax.set_title('Accuracy on %s Set' % imset)

blue_patch = Patch(color='b', label='Correct Accuracy')
red_patch = Patch(color='r', label='Amount Overconfident')
orange_patch = Patch(color='tab:orange', label='Amount Underconfident')
ax.legend(handles=[blue_patch, red_patch, orange_patch])

fig.savefig('images/acc_hist.png', bbox_inches='tight')
plt.show()