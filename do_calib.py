import numpy as np
from os.path import join, isfile, isdir
import os
import shutil
import util

def calibrate_sm_for_idxs(idxs, slices, args):
	for idx in idxs:
		lgt_vol = util.load_lgt_aug(args.imset, idx)
		num_aug, h, w, nc = lgt_vol.shape

		gt = util.load_gt(args.imset, idx)

		fgmask = util.fg_mask_for(gt)
		fgy, fgx = np.where(fgmask)

		lgt_vol = lgt_vol[:,fgy,fgx,:].reshape(num_aug, -1, nc)
		gt = gt[fgmask].ravel()

		sm = util.sm_of_logits(lgts)

		for slc in slices:
			sm = slc.remap_sm(sm)
			avg_sm = sm.mean(0)

			preds = np.argmax(avg_sm, -1)
			pred_sm = sm[...,preds]

			min_pred_sm = pred_sm.min(0)
			max_pred_sm = pred_sm.max(0)

			slc.remap_labels(gt)

			for i in np.unique(preds):
				pred_mask = preds == i
				gt_masked = gt[pred_mask]

				min_sm_masked = min_pred_sm[pred_mask]
				max_sm_masked = max_pred_sm[pred_mask]

				slc[i].accum_sm(min_sm_masked, max_sm_masked, gt_masked == i)

	return slices

def calibrate_sm_for_idxs_unpack(params):
	return calibrate_sm_for_idxs(*params)

def aggregate_proc_confs(proc_slices, slices, args):
	for i, slc in enumerate(slices):
		for j, main_node in enumerate(slc):
			main_node.__init__(main_node.name, main_node.node_idx, main_node.children, main_node.nb, data_dir=args.data_dir, is_main=True)

			for proc_slice in proc_slices:
				proc_node = proc_slice[i][j]
				main_node.accum_node(proc_node)

			main_node.save()

	return slices


if __name__ == '__main__':
	from argparse import ArgumentParser
	parser = ArgumentParser(description='Build the calibration hierarchy using multiprocessing.')
	parser.add_argument('--slice_file', dest='slice_file', type=str, default='slices.pkl', help='The pickle file that specifies the hierarchy.')
	parser.add_argument('--imset', dest='imset', type=str, default='val', help='The image set to build the calibration confograms from. Either val or test')
	parser.add_argument('--num_proc', dest='num_proc', type=int, default=8, help='The number of processes to spawn to parallelize calibration.')
	parser.add_argument('--nb', dest='nb', type=int, default=100, help='The number of bins in the calibration histogram.')
	parser.add_argument('--output_file', dest='output_file', type=str, default=None, help='The pickle file to output the calibration hierarchy to. None if slice_file to be overwritten.')
	parser.add_argument('--data_dir', dest='data_dir', type=str, default='calib_data', help='The data to store confidences in')
	parser.add_argument('--test', dest='test', action='store_true', help='Whether or not to test the calibration script. Takes the first 2*num_proc from the imset.')
	args = parser.parse_args()
	
	if not isdir(args.data_dir):
		os.mkdir(args.data_dir)

	slices = util.read_slices(args.slice_file)
	for slc in slices:
		for node in slc:
			node.__init__(node.name, node.node_idx, node.children, node.nb, data_dir=args.data_dir)

	idxs = list(range(1, util.num_img_for(args.imset)+1))
	if args.test:
		idxs = idxs[:2*args.num_proc]

	param_batches = [(idxs[i::args.num_proc], slices.copy(), args) for i in range(args.num_proc)] 

	with util.poolcontext(args.num_proc) as p:
		proc_slices = p.map(calibrate_sm_for_idxs_unpack, param_batches)

	main_slices = slices.copy()
	main_slices = aggregate_proc_confs(proc_slices, main_slices, args)

	output_fname = args.output_file
	if output_fname is None:
		output_fname = args.slice_file

	util.save_slices(output_fname, main_slices)
	
	if args.test:
		shutil.rmtree(args.data_dir)
