from util import *
from hdf5storage import savemat
import os
from os.path import join, isdir

pred_path = join(ds_path, 'deeplab_prediction', 'test', 'dl_pred')
os.mkdir(pred_path)

for idx in range(1, num_img_for('test')+1):
	stdout_writeln(str(idx))

	logits = load_logits('test', idx)
	gt = load_gt('test', idx)
	fg_mask = fg_mask_for(gt)

	tot_pred = np.zeros_like(gt)
	
	fg_logits = logits[fg_mask]
	fg_pred = np.argmax(fg_logits[...,1:], axis=-1) + 1

	tot_pred[fg_mask] = fg_pred

	savemat(join(pred_path, 'test_%06d_prediction.mat' % idx), {'pred_img': tot_pred})