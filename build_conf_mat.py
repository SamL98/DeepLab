import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join, isfile

DS_PATH = 'D:/datasets/processed/voc2012' 
ds_info = loadmat(join(DS_PATH, 'dataset_info.mat'))

if not isfile('confusion_matrix.mat'):
	import sys
	imset = sys.argv[1].lower().capitalize()

	PRED_PATH = join(DS_PATH, 'Deeplab_Prediction', imset)
	GT_PATH = join(DS_PATH, 'Truth', imset)

	num_img = ds_info['num_'+imset.lower()]
	num_classes = 20
	bg_class = 0

	LOGIT_FMT = imset.lower()+'_%06d_logits.mat'
	LOGIT_MAT_NAME = 'logits_img'

	GT_FMT = imset.lower()+'_%06d_pixeltruth.mat'
	GT_MAT_NAME = 'truth_img'

	conf_mat = np.zeros((num_classes, num_classes), dtype=np.uint64)

	for idx in range(1, num_img+1):
		logit_im = loadmat(join(PRED_PATH, LOGIT_FMT % idx))[LOGIT_MAT_NAME][...,1:].reshape(-1, num_classes)
		gt = loadmat(join(GT_PATH, GT_FMT % idx))[GT_MAT_NAME].flatten()
	
		for tr, logits in zip(gt, logit_im):
			if tr == bg_class or tr == 255: continue
			conf_mat[np.argmax(logits), tr-1] += 1
		
	savemat('confusion_matrix.mat', {'confusion_matrix': conf_mat})
else:
	conf_mat = loadmat('confusion_matrix.mat')['confusion_matrix']
	
	
conf_mat[range(20), range(20)] = 0
conf_mat = conf_mat.astype(np.float32)/conf_mat.sum()

class_labels = ds_info['class_labels'][1:]
class_labels[class_labels.index('aeroplane')] = 'plane'
class_labels[class_labels.index('diningtable')] = 'table'
class_labels[class_labels.index('motorbike')] = 'mbike'
class_labels[class_labels.index('potted plant')] = 'plant'
class_labels[class_labels.index('tvmonitor')] = 'tv'

import matplotlib.pyplot as plt
import seaborn as sn

plt.rcParams['figure.figsize'] = [12, 10]

fig, ax = plt.subplots(1)
sn.heatmap(conf_mat, annot=True, ax=ax, fmt='.2%', annot_kws={'fontsize': 8})

ax.set_title('Confusion Matrix')
ax.set_xticklabels(class_labels, rotation=-80)
ax.set_yticklabels(class_labels, rotation=0)
ax.set_xlabel('True')
ax.set_ylabel('Pred')

fig.savefig('confusion_matrix.png')#, dpi=800)
plt.show()