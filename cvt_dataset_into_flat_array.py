import util
from os.path import join
import sys
import numpy as np

if __name__ == '__main__':
	imset = sys.argv[1]
	n_proc = int(sys.argv[2])
	n_img = util.num_img_for(imset)

	idxs = range(1, n_img+1)
	for procno in range(n_proc):
		proc_idxs = idxs[procno::n_proc]
		for idx in proc_idxs:
			logits = util.load_logits(imset, idx).astype(np.float64)
			gt = util.load_gt(imset, idx)
			util.serialize_lgt_gt_pair(logits, gt, imset, procno)
