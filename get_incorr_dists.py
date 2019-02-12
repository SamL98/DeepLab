import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join, isfile
from scipy.ndimage.morphology import distance_transform_edt as distance_transform

DS_PATH = 'D:/datasets/processed/voc2012' 
ds_info = loadmat(join(DS_PATH, 'dataset_info.mat'))

if not isfile('incorr_dist_hist.mat'):
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

	res = .5
	nb = int(50./res)
	dist_hist = np.zeros((nb), dtype=np.uint64)

	for idx in range(1, num_img+1):
		logit_im = loadmat(join(PRED_PATH, LOGIT_FMT % idx))[LOGIT_MAT_NAME][...,1:].reshape(-1, num_classes)
		gt = loadmat(join(GT_PATH, GT_FMT % idx))[GT_MAT_NAME]
		
		fg = np.ones_like(gt)
		fg[(gt==0) | (gt==255)] = 0
		#weightmap = np.minimum(20, distance_transform(fg)).ravel()/20.
		weightmap = distance_transform(fg).ravel()
	
		for i, (tr, logits) in enumerate(zip(gt.ravel(), logit_im)):
			if tr == bg_class or tr == 255: continue
			
			pred = np.argmax(logits)
			if tr-1 == pred: continue
			
			dist_hist[min(int(np.floor(min(weightmap[i], 50)/res)), nb-1)] += 1
		
	savemat('incorr_dist_hist.mat', {'dist_hist': dist_hist})
else:
	dist_hist = loadmat('incorr_dist_hist.mat')['dist_hist']
	
	
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)

ax.bar(range(len(dist_hist)), dist_hist)

fig.savefig('incorr_dist_hist.png')#, dpi=800)
plt.show()