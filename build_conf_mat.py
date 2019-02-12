import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join, isfile
from scipy.ndimage.morphology import distance_transform_edt as distance_transform

DS_PATH = 'D:/datasets/processed/voc2012' 
ds_info = loadmat(join(DS_PATH, 'dataset_info.mat'))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--imset', type=str, default='val', dest='imset')
parser.add_argument('--use_dt', action='store_true', dest='use_dt')
parser.add_argument('--pv', type=int, default=20, dest='plateau_val')
parser.add_argument('--include_test', action='store_true', dest='include_test')
args = parser.parse_args()

imset = args.imset.lower().capitalize()

def make_fnames(args):
	fname = 'confusion_matrix'
	if args.use_dt:
		fname += '_dist'
		if not args.plateau_val is None:
			fname += '_pv%d' % args.plateau_val
			
	if args.include_test:
		fname += '_test'
		
	return join('mats', fname+'.mat'), join('images', fname+'.png')
	
mat_fname, img_fname = make_fnames(args)


def add_imset_to_conf_mat(imset, conf_mat):
	global DS_PATH, args

	PRED_PATH = join(DS_PATH, 'Deeplab_Prediction', imset)
	GT_PATH = join(DS_PATH, 'Truth', imset)
	
	if imset.lower() == 'val':
		num_img = 350
	else:
		num_img = 1099
		
	num_classes = 20
	bg_class = 0

	LOGIT_FMT = imset.lower()+'_%06d_logits.mat'
	LOGIT_MAT_NAME = 'logits_img'

	GT_FMT = imset.lower()+'_%06d_pixeltruth.mat'
	GT_MAT_NAME = 'truth_img'
	
	for idx in range(1, num_img+1):
		logit_im = loadmat(join(PRED_PATH, LOGIT_FMT % idx))[LOGIT_MAT_NAME][...,1:].reshape(-1, num_classes)
		gt = loadmat(join(GT_PATH, GT_FMT % idx))[GT_MAT_NAME]
		
		if not args.use_dt:
			weightmap = np.ones((len(logit_im)), dtype=np.uint8)
		else:
			fg = np.ones_like(gt)
			fg[(gt==0) | (gt==255)] = 0
			weightmap = distance_transform(fg).ravel()
			
			if not args.plateau_val is None:
				weightmap = np.minimum(args.plateau_val, weightmap)
	
		for i, (tr, logits) in enumerate(zip(gt.ravel(), logit_im)):
			if tr == bg_class or tr == 255: continue
			conf_mat[np.argmax(logits), tr-1] += weightmap[i]
			
	return conf_mat
	

if not isfile(mat_fname):
	conf_mat = np.zeros((num_classes, num_classes), dtype=np.uint64)
	conf_mat = add_imset_to_conf_mat(imset, conf_mat)
	
	if imset.lower() == 'val' and args.include_test:
		conf_mat = add_imset_to_conf_mat('test', conf_mat)
		
	savemat(mat_fname, {'confusion_matrix': conf_mat})
else:
	conf_mat = loadmat(mat_fname)['confusion_matrix']
	
	
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
sn.heatmap(conf_mat, ax=ax, annot=True, fmt='.2%', annot_kws={'fontsize': 8})

ax.set_title('Confusion Matrix')
ax.set_xticklabels(class_labels, rotation=-80)
ax.set_yticklabels(class_labels, rotation=0)
ax.set_xlabel('True')
ax.set_ylabel('Pred')

fig.savefig(img_fname)#, dpi=800)
plt.show()