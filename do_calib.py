import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join, isfile, isdir
import os

from util import *

# Get the confograms for the given logits and ground truth labels
def confs_for_pixels(logits, gt, slices, args):
	terminal_pred = np.argmax(logits[:,1:], axis=-1) + 1

	# If we are not taking the softmax by slice, take the softmax once and be done with it
	if not args.sm_by_slice:
		sm = sm_of_logits(logits, start_idx=1, zero_pad=True)

	for i, slc in enumerate(slices):
		# Remap the terminal predictions to the local labels of the current slice
		slc_gt = np.array([remap_label(lab, slc) for lab in gt])

		# Remap the ground truth to the local labels of the current slice
		slc_term_pred = np.array([remap_label(pred, slc) for pred in terminal_pred])

		if args.sm_by_slice:
			slc_logits = np.array([remap_scores(logit_vec, slc) for logit_vec in logits])
			slc_sm = sm_of_logits(slc_logits)
		else:
			slc_sm = np.array([remap_scores(sm_vec, slc) for sm_vec in sm])

		for j, node in enumerate(slc):
			# Create a mask of where the terminal prediction was this label
			pred_mask = slc_term_pred == j

			slc_gt_masked = slc_gt[pred_mask]
			slc_sm_masked = slc_sm[pred_mask]

			# Because of the previous mask, the j-th softmax value will always be the predicted softmax score
			sm_conf = slc_sm_masked[:,j]

			# Save the confidence of each pixel as well as whether it was correct to disk
			node.accum_scores(sm_conf, slc_gt_masked == j, args.nb, args.sigma)

	return slices


# Return the correct and count confograms given the hierarchy specified by slices
def get_confs_for_idxs(idxs, slices, args):
	for slc in slices:
		for node in slc:
			node.__init__(node.name, node.node_idx, node.terminals, data_dir=args.data_dir)

	# If we are computing the confograms on the fly, load each image individually and accumulate all the confograms
	for idx in idxs:
		logits = load_logits(args.imset, idx, reshape=True)
		gt = load_gt(args.imset, idx, reshape=True)

		fg_mask = fg_mask_for(gt)
		logits = logits[fg_mask]
		gt = gt[fg_mask]

		slices = confs_for_pixels(logits, gt, slices, args)

	return slices

def get_confs_for_idxs_unpack(params):
	return get_confs_for_idxs(*params)

def aggregate_proc_confs(proc_slices, slices, args):
	for i, slc in enumerate(slices):
		for j, node in enumerate(slc):
			node.__init__(node.name, node.node_idx, node.terminals, data_dir=args.data_dir, is_main=True)
			node.reset(args.nb)

			for proc_slice in proc_slices:
				proc_node = proc_slice[i][j]
				
				if not hasattr(proc_node, 'tot_hist'):
					continue

				node.accum_node(proc_node)

			node.generate_acc_hist(args.nb, args.alpha)

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
