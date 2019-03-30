import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join, isfile, isdir
import os
import shutil

import util

def calibrate_sm_for_chunk(chunkno, chunklen, slices, args):
	batch_size = 10000 # calculate later	
	batch_sizes = [batch_size] * (chunklen//batch_size)
	batch_sizes[-1] -= chunklen % batch_size

	for batch in batch_sizes:
		_, _, _, lgts, gt = util.unserialize_examples(args.imset, batch, chunkno) 	
		lgts = lgts.reshape(-1, util.nc)
		term_preds = np.argmax(lgts, -1) + 1

		if not args.sm_by_slice: scores = util.sm_of_logits(logits, zero_pad=True)
		else: scores = logits

		for slc in slices:
			slc_gt = util.remap_label_arr(gt, slc)
			slc_term_pred = util.remap_label_arr(term_pred, slc)
			slc_scores = util.remap_scores_arr(scores, slc)

			if args.sm_by_slice: slc_sm = util.sm_of_logits(slc_scores)
			else: slc_sm = slc_scores

			for i in np.unique(slc_term_pred):
				pred_mask = slc_term_pred == i
				slc_gt_masked = slc_gt[pred_mask]
				slc_sm_masked = slc_sm[pred_mask]

				slc_sm_val = slc_sm_masked[:,i]
				node.accum_scores(slc_sm_val, slc_gt_masked == i, args.nb, args.sigma)

	return slices

def calibrate_sm_for_chunk_unpack(params):
	return calibrate_sm_for_chunk(*params)

def aggregate_proc_confs(proc_slices, slices, args):
	for i, slc in enumerate(slices):
		for j, main_node in enumerate(slc):
			main_node.__init__(main_node.name, main_node.node_idx, main_node.terminals, data_dir=args.data_dir, is_main=True)
			main_node.reset(args.nb)

			for proc_slice in proc_slices:
				proc_node = proc_slice[i][j]

				if not hasattr(proc_node, 'n_c'):
					continue

				main_node.accum_node(proc_node)

			main_node.generate_acc_hist(args.nb, args.alpha)

	return slices


if __name__ == '__main__':
	from argparse import ArgumentParser
	parser = ArgumentParser(description='Build the calibration hierarchy using multiprocessing.')
	parser.add_argument('--slice_file', dest='slice_file', type=str, default='slices.pkl', help='The pickle file that specifies the hierarchy.')
	parser.add_argument('--imset', dest='imset', type=str, default='val', help='The image set to build the calibration confograms from. Either val or test')
	parser.add_argument('--num_proc', dest='num_proc', type=int, default=8, help='The number of processes to spawn to parallelize calibration.')
	parser.add_argument('--sigma', dest='sigma', type=float, default=0.1, help='The bandwidth for parzen estimation.')
	parser.add_argument('--alpha', dest='alpha', type=float, default=0.05, help='The confidence for the wilson interval.')
	parser.add_argument('--nb', dest='nb', type=int, default=100, help='The number of bins in the calibration histogram.')
	parser.add_argument('--output_file', dest='output_file', type=str, default=None, help='The pickle file to output the calibration hierarchy to. None if slice_file to be overwritten.')
	parser.add_argument('--sm_by_slice', dest='sm_by_slice', action='store_true', help='Whether or not to take the softmax of the logits at each slice of the hierarchy. True by default.')
	parser.add_argument('--data_dir', dest='data_dir', type=str, default='calib_data', help='The data to store confidences in')
	parser.add_argument('--test', dest='test', action='store_true', help='Whether or not to test the calibration script. Takes the first 2*num_proc from the imset.')
	args = parser.parse_args()
	
	if not isdir(args.data_dir):
		os.mkdir(args.data_dir)

	slices = util.read_slices(args.slice_file, reset=args.reset)

	n_img = util.num_img_for(args.imset)
	img_per_chunk = [n_img//args.num_proc] * args.num_proc
	img_per_chunk[-1] -= n_img % args.num_proc

	param_batches = [(i, chnk_len, slices.copy(), args) for i, chnk_len in zip(range(args.num_proc), img_per_chunk)] 

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
