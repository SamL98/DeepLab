import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join, isfile, isdir
import os
import shutil

import util

def calibrate_sm_for_chunk(chunkno, slices, args):
	batch_size = 500000 # calculate later	

	lgts = np.zeros((batch_size, util.nc), dtype=util.DTYPES[util.LOGITS])
	gt = np.zeros((batch_size), dtype=util.DTYPES[util.GT])

	done = False

	while not done:
		done, num_pix = util.unserialize_examples_for_calib(args.imset, batch_size, chunkno, lgts, gt) 	
		term_preds = np.argmax(lgts, -1)
		gt -= 1

		if done:
			lgts = lgts[:num_pix]
			gt = gt[:num_pix]
			term_preds = term_preds[:num_pix]

		scores = util.sm_of_logits(lgts)

		for slc in slices:
			scores = slc.remap_scores_and_labels(scores, gt, term_preds)
			sm = scores

			for i in np.unique(term_preds):
				pred_mask = term_preds == i
				gt_masked = gt[pred_mask]
				sm_masked = sm[pred_mask]

				sm_val = sm_masked[:,i]
				node = slc[i]
				node.accum_scores(sm_val, gt_masked == i, args.nb, args.sigma)

	return slices

def calibrate_sm_for_chunk_unpack(params):
	return calibrate_sm_for_chunk(*params)

def aggregate_proc_confs(proc_slices, slices, args):
	for i, slc in enumerate(slices):
		for j, main_node in enumerate(slc):
			main_node.__init__(main_node.name, main_node.node_idx, main_node.children, data_dir=args.data_dir, is_main=True)
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
	parser.add_argument('--data_dir', dest='data_dir', type=str, default='calib_data', help='The data to store confidences in')
	parser.add_argument('--test', dest='test', action='store_true', help='Whether or not to test the calibration script. Takes the first 2*num_proc from the imset.')
	args = parser.parse_args()
	
	if not isdir(args.data_dir):
		os.mkdir(args.data_dir)

	slices = util.read_slices(args.slice_file)

	n_img = util.num_img_for(args.imset)

	param_batches = [(i, slices.copy(), args) for i in range(args.num_proc)] 

	with util.poolcontext(args.num_proc) as p:
		proc_slices = p.map(calibrate_sm_for_chunk_unpack, param_batches)

	main_slices = slices.copy()
	main_slices = aggregate_proc_confs(proc_slices, main_slices, args)

	output_fname = args.output_file
	if output_fname is None:
		output_fname = args.slice_file

	util.save_slices(output_fname, main_slices)
	
	if args.test:
		shutil.rmtree(args.data_dir)
