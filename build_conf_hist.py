import numpy as np
from hdf5storage import loadmat
from os.path import join
	
import sys
imset = sys.argv[1].lower().capitalize()

DS_PATH = 'D:/datasets/processed/voc2012' 
GT_PATH = join(DS_PATH, 'Truth', imset)
PRED_PATH = join(DS_PATH, 'Deeplab_Prediction', imset)

ds_info = loadmat(join(DS_PATH, 'dataset_info.mat'))
num_img = 350

if imset.lower() != 'val':
	num_img = 1449-num_img

LGT_FMT = join(PRED_PATH, imset.lower()+'_%06d_logits.mat')
LGT_MAT_NAME = 'logits_img'

GT_FMT = join(GT_PATH, imset.lower()+'_%06d_pixeltruth.mat')
GT_MAT_NAME = 'truth_img'
	
conf_res = 0.05
conf_hist = np.zeros((int(1./conf_res)), dtype=np.uint64)
	
for idx in range(1, num_img+1):
	gt = loadmat(GT_FMT % idx)[GT_MAT_NAME].ravel()
	mask = (gt>0) & (gt<255)

	logits = loadmat(LGT_FMT % idx)[LGT_MAT_NAME][...,1:].reshape(-1, 20)
	logits = logits[mask]
	
	exp_logits = np.exp(logits)
	sm = exp_logits / np.maximum(1e-7, exp_logits.sum(-1)[:,np.newaxis])
	scores = np.max(sm, axis=-1)
	
	bins = np.floor(scores/conf_res).astype(np.uint8)
	
	for binno in np.unique(bins):
		conf_hist[min(binno, conf_hist.shape[0]-1)] += np.sum(bins==binno)

conf_hist = conf_hist.astype(np.float64)/conf_hist.sum()
		
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(range(conf_hist.shape[0]), conf_hist, align='edge')
ax.set_xlabel('Confidence')
ax.xaxis.set_ticks(np.linspace(0, conf_hist.shape[0], num=6))
ax.set_xticklabels(['%.1f' % v for v in np.linspace(0.0, 1.0, num=6)])
ax.set_ylabel('% of pixels')
ax.yaxis.set_ticks(np.linspace(0.0, 1.0, num=6))
ax.set_title('Confidence on %s Set' % imset)
fig.savefig('images/conf_hist.png', bbox_inches='tight')
plt.show()