import numpy as np
from hdf5storage import loadmat
from os.path import join
	
import sys
imset = sys.argv[1].lower().capitalize()

DS_PATH = 'D:/datasets/processed/voc2012' 
PRED_PATH = join(DS_PATH, 'Deeplab_Prediction', imset)

ds_info = loadmat(join(DS_PATH, 'dataset_info.mat'))
num_img = ds_info['num_'+imset.lower()]

SM_FMT = '%s_%06d_softmax.mat'
SM_MAT_NAME = 'softmax_img'

def load_softmax(idx):
	global PRED_PATH, SM_FMT, SM_MAT_NAME, imset
	return loadmat(join(PRED_PATH, SM_FMT % (imset, idx)))[SM_MAT_NAME][:]
	
	
conf_res = 0.05
conf_hist = np.zeros((int(1./conf_res)), dtype=np.uint64)
#count_hist = np.zeros_like(conf_hist)
	
for i in range(1, num_img+1):
	scores = load_softmax(i).flatten()
	bins = np.floor(scores/conf_res).astype(np.uint8)
	
	for binno in np.unique(bins):
		conf_hist[min(binno, conf_hist.shape[0]-1)] += np.sum(bins==binno)
		#conf_hist[min(binno, conf_hist.shape[0]-1)] += np.sum(scores[bins==binno])
		#count_hist[min(binno, count_hist.shape[0]-1)] += np.sum(bins==binno)

		
#conf_hist = conf_hist.astype(np.float64)/np.maximum(1e-7, count_hist.astype(np.float64))
conf_hist = conf_hist.astype(np.float64)/conf_hist.sum()
		
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(range(conf_hist.shape[0]), conf_hist, align='edge')
ax.set_xlabel('Confidence')
ax.xaxis.set_ticks(np.linspace(0, conf_hist.shape[0], num=6))
ax.set_xticklabels(['%.1f' % v for v in np.linspace(0.0, 1.0, num=6)])
ax.set_ylabel('% of pixels')
ax.yaxis.set_ticks(np.linspace(0.0, 1.0, num=6))
ax.set_title('%s Set' % imset)
fig.savefig('conf_hist.png', bbox_inches='tight')
plt.show()