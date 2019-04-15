import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join 
import sys

import util

def perform_inference_on_idxs(idxs, slices, args):
	for idx in idxs:
		lgt_vol = util.load_lgt_aug(args.imset, idx)
		num_aug, h, w, nc = lgt_vol.shape

		gt = util.load_gt(args.imset, idx)

		fgmask = util.fg_mask_for(gt)
		fgy, fgx = np.where(fgmask)

		lgt_vol = lgt_vol[:,fgy,fgx,:].reshape(num_aug, -1, nc)
		gt = gt[fgmask].ravel()

		sm = util.sm_of_logits(lgts)

		per_slice_preds = []
		per_slice_confs = []

		for j, slc in enumerate(slices):
			sm = slc.remap_sm(sm)
			avg_sm = sm.mean(0)

			preds = np.argmax(avg_sm, -1)
			pred_sm = sm[...,preds]

			min_pred_sm = pred_sm.min(0)
			max_pred_sm = pred_sm.max(0)

			slc_conf_mask = np.zeros_like(preds)
			slc_conf_map = np.zeros_like(pred_sm)

			for slc_pred_lab in np.unique(preds):
				node = slc[slc_pred_lab]

				pred_mask = preds == slc_pred_lab	
				confs = node.confs_for_sm(min_pred_sm[pred_mask], max_pred_sm[pred_mask])

				pred_lab = node.node_idx
				slice_idx = j
				while slice_idx > 0 and len(node.children) == 1:
					pred_lab = node.children[0]
					node = slices[slice_idx-1][pred_lab]
					slice_idx -= 1

				slc_conf_mask[pred_mask] = pred_lab
				slc_conf_map[pred_mask] = confs

			per_slice_preds.append(util.set_fg_in_larger_array(slc_conf_mask, fgmask, (h, w)))
			per_slice_confs.append(util.set_fg_in_larger_array(slc_conf_map, fgmask, (h, w)))

		util.save_calib_pred(args.imset, idx, args.name, per_slice_preds, per_slice_confs)
	
def perform_inference_on_idxs_unpack(params):
	perform_inference_on_idxs(*params)
	
if __name__ == '__main__':
	from argparse import ArgumentParser
	parser = ArgumentParser(description='Build the calibration hierarchy using multiprocessing.')
	parser.add_argument('--slice_file', dest='slice_file', type=str, default='slices.pkl', help='The pickle file that specifies the hierarchy.')
	parser.add_argument('--imset', dest='imset', type=str, default='test', help='The image set to build the calibration histograms from. Either val or test')
	parser.add_argument('--num_proc', dest='num_proc', type=int, default=8, help='The number of processes to spawn to parallelize inference.')
	parser.add_argument('--name', dest='name', type=str, help='The name of the current method.')
	args = parser.parse_args()

	slices = util.read_slices(args.slice_file)

	idxs = list(range(1, util.num_img_for(args.imset)+1))
	param_batches = [(idxs[i::args.num_proc], slices.copy(), args) for i in range(args.num_proc)] 

	with util.poolcontext(args.num_proc) as p:
		_ = p.map(perform_inference_on_idxs_unpack, param_batches)
