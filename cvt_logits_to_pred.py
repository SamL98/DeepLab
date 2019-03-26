from util import *
from hdf5storage import savemat
import os
from os.path import join, isdir

pred_path = join(ds_path, 'deeplab_prediction', 'test', 'dl_pred')
os.mkdir(pred_path)

for idx in range(1, num_img_for('test')+1):
	stdout_writeln(str(idx))
	logits = load_logits('test', idx)
	pred = np.argmax(logits, -1)
	savemat(join(pred_path, 'test_%06d_prediction.mat' % idx), {'pred_img': pred})