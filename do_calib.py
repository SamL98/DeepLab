import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join, isfile, isdir
import os

from util import *

def calibrate_sm_for_image(idx, slices, args):
	'''
	Calibrates the softmax value for one (logits, ground truth) pair

	Params:
		idx: The index of the image within args.imset
		slices: A copy of the original slices
		args: The command line arguments

	Returns:
		A copy of the slices with the single-image counts accumulated
	'''

	logits, term_pred, gt = load_logits_pred_gt_triplet(args.imset, idx)

	if not args.sm_by_slice: scores = sm_of_logits(logits, start_idx=1, zero_pad=True)
	else: scores = logits

	for slc in slices:
		slc_gt = remap_label_arr(gt, slc)
		slc_term_pred = remap_label_arr(gt, slc)
		slc_scores = remap_scores_arr(scores, slc)

		if args.sm_by_slice: slc_sm = sm_of_logits(slc_score)
		else: slc_sm = slc_scores

		for i, node in enumerate(slc):
			pred_mask = slc_term_pred == i
			slc_gt_masked = slc_gt[pred_mask]
			slc_sm_masked = slc_sm[pred_mask]

			slc_sm_val = slc_sm_masked[:,i]
			node.accum_scores(slc_sm_val, slc_gt_masked == i, args.nb, args.sigma)

	return slices


def calibrate_sm_for_idxs(idxs, slices, args):
	'''
	Calibrates the DeepLab-predicted softmax values for the given indices

	Params:
		idxs: An array of indices into the given imset to calibrate
		slices: A copy of the original slices
		args: The command-line arguments passed to the script

	Returns:
		A copy of the original slices with count distributions accumulated
	'''
	for slc in slices:
		for node in slc:
			node.__init__(node.name, node.node_idx, node.terminals, data_dir=args.data_dir)

	for idx in idxs:
		slices = calibrate_sm_for_image(idx, slices, args)

	return slices

def calibrate_sm_for_idxs_unpack(params):
	'''
	Wrapper for calibrate_sm_for_idxs
	'''
	return calibrate_sm_for_idxs(*params)

def aggregate_proc_confs(proc_slices, slices, args):
	for i, slc in enumerate(slices):
		for j, main_node in enumerate(slc):
			main_node.__init__(main_node.name, main_node.node_idx, main_node.terminals, data_dir=args.data_dir, is_main=True)
			main_node.reset(args.nb)

			for proc_slice in proc_slices:
				proc_node = proc_slice[i][j]

				if not hasattr(proc_node, 'tot_hist'):
					continue

				main_node.accum_node(proc_node)

			main_node.generate_acc_hist(args.nb, args.alpha)

	return slices


from argparse import ArgumentParser
parser = ArgumentParser(description='Build the calibration hierarchy using multiprocessing.')
parser.add_argument('--slice_file', dest='slice_file', type=str, default='slices.pkl', help='The pickle file that specifies the hierarchy.')
parser.add_argument('--imset', dest='imset', type=str, default='val', help='The image set to build the calibration confograms from. Either val or test')
parser.add_argument('--num_proc', dest='num_proc', type=int, default=8, help='The number of processes to spawn to parallelize calibration.')
parser.add_argument('--sigma', dest='sigma', type=float, default=0.1, help='The bandwidth for parzen estimation.')
parser.add_argument('--alpha', dest='alpha', type=float, default=0.05, help='The confidence for the wilson interval.')
parser.add_argument('--nb', dest='nb', type=int, default=100, help='The number of bins in the calibration histogram.')
parser.add_argument('--output_file', dest='output_file', type=str, default=None, help='The pickle file to output the calibration hierarchy to. None if slice_file to be overwritten.')
parser.add_argument('--dont_reset', dest='reset', action='store_false', help='Pass if you want to accumulate calibration confograms. Normally they are reset when this script is run.')
parser.add_argument('--sm_by_slice', dest='sm_by_slice', action='store_true', help='Whether or not to take the softmax of the logits at each slice of the hierarchy. True by default.')
parser.add_argument('--data_dir', dest='data_dir', type=str, default='calib_data', help='The data to store confidences in')
parser.add_argument('--test', dest='test', action='store_true', help='Whether or not to test the calibration script. Takes the first 2*num_proc from the imset.')

if __name__ == '__main__':
	args = parser.parse_args()
	
	if not isdir(args.data_dir):
		os.mkdir(args.data_dir)

	slices = read_slices(args.slice_file, reset=args.reset)
	param_batches = get_param_batches(slices, args)

	with poolcontext(args.num_proc) as p:
		proc_slices = p.map(get_confs_for_idxs_unpack, param_batches)

	main_slices = slices.copy()
	main_slices = aggregate_proc_confs(proc_slices, main_slices, args)

	output_fname = args.output_file
	if output_fname is None:
		output_fname = args.slice_file

	save_slices(output_fname, main_slices)
	
	if args.test:
		os.remove(args.data_dir)
