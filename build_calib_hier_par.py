import numpy as np
import multiprocessing as mp
from contextlib import contextmanager
from hdf5storage import loadmat, savemat
from os.path import join, isfile, isdir
import os

from util import *

# Pool the a function across multiple inputs and wait for them to complete
@contextmanager
def poolcontext(num_proc):
    pool = mp.Pool(num_proc)
    yield pool
    pool.terminate()

# Get the confograms for the given logits and ground truth labels
def confs_for_pixels(logits, gt, slices, args):
	# If we are not taking the softmax by slice, take the softmax once and be done with it
	if not args.sm_by_slice:
		sm = sm_of_logits(logits, start_idx=1, zero_pad=True)

	for i, slc in enumerate(slices):
		# Remap the ground truth to the local labels of the current slice
		slc_gt = np.array([remap_gt(lab, slc) for lab in gt])
		
		# If we are taking the softmax by slice, remap the logits then take the softmax
		#
		# Otherwise, just remap the softmax previously computed
		if args.sm_by_slice:
			slc_logits = np.array([remap_scores(logit_vec, slc) for logit_vec in logits])
			slc_sm = sm_of_logits(slc_logits)
		else:
			slc_sm = np.array([remap_scores(sm_vec, slc) for sm_vec in sm])

			
		for j, node in enumerate(slc):
			# Since we are measuring precision in the calibration confograms, mask the ground truth and softmax by where the current node is softmax
			pred_labels = np.argmax(slc_sm, axis=-1)
			argmax_mask = pred_labels == j

			slc_gt_masked = slc_gt[argmax_mask]
			slc_sm_masked = slc_sm[argmax_mask]

			# Because of the previous mask, the j-th softmax value will always be the max
			sm_conf = slc_sm_masked[:,j]

			# Save the confidence of each pixel as well as whether it was correct to disk
			node.append_confs(sm_conf, slc_gt_masked == j)

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

def aggregate_proc_confs(proc_slices, slices, data_dir):
	for i, slc in enumerate(slices):
		for j, node in enumerate(slc):
			node.__init__(node.name, node.node_idx, node.terminals, data_dir=args.data_dir, is_main=True)
		
			conf_f = open(data_dir + '/%s_confs.txt' % node.name, 'w')
			corr_f = open(data_dir + '/%s_corr.txt' % node.name, 'w')

			for proc_slice in proc_slices:
				proc_node = proc_slice[i][j]
				
				conf, corr_mask = proc_node.get_file_contents()
				proc_node.reset()
				
				if conf is None:
					continue

				np.savetxt(conf_f, conf)
				np.savetxt(corr_f, corr_mask)

			conf_f.close()
			corr_f.close()

	return slices


from argparse import ArgumentParser
parser = ArgumentParser(description='Build the calibration hierarchy using multiprocessing.')
parser.add_argument('--slice_file', dest='slice_file', type=str, default='slices.pkl', help='The pickle file that specifies the hierarchy.')
parser.add_argument('--imset', dest='imset', type=str, default='val', help='The image set to build the calibration confograms from. Either val or test')
parser.add_argument('--num_proc', dest='num_proc', type=int, default=1, help='The number of processes to spawn to parallelize calibration.')
parser.add_argument('--output_file', dest='output_file', type=str, default=None, help='The pickle file to output the calibration hierarchy to. None if slice_file to be overwritten.')
parser.add_argument('--dont_reset', dest='reset', action='store_false', help='Pass if you want to accumulate calibration confograms. Normally they are reset when this script is run.')
parser.add_argument('--sm_by_slice', dest='sm_by_slice', action='store_true', help='Whether or not to take the softmax of the logits at each slice of the hierarchy. True by default.')
parser.add_argument('--data_dir', dest='data_dir', type=str, default='calib_data', help='The data to store confidences in')
parser.add_argument('--test', dest='test', action='store_true', help='Whether or not to test the calibration script. Takes the first 2*num_proc from the imset.')


if __name__ == '__main__':
	args = parser.parse_args()
	
	if not isdir(args.data_dir):
		os.mkdir(args.data_dir)

	# Load the slices from the specified file

	slices = read_slices(args.slice_file, reset=args.reset)

	# Load the index ordering -- indexes are ordered by number of foreground pixels in descending order
	#
	# This way, if multiprocessing is used, all processes will be given approximately the same workload

	idx_ordering = None
	idx_ordering_fname = args.imset.lower() + '_ordered.txt'

	if not isfile(idx_ordering_fname):
		from order_by_num_fg import order_imset_by_num_fg
		idx_ordering = order_imset_by_num_fg(args.imset, save=True)
	else:
		with open(idx_ordering_fname) as f:
			idx_ordering = [int(idx) for idx in f.read().split('\n')]
			
			
	if args.test:
		idx_ordering = idx_ordering[:args.num_proc]

	# Split the indexes up between processed to try and spread the work evenly

	idx_ordering = np.array(idx_ordering)
	param_batches = []

	for procno in range(args.num_proc):
		idx_batch = idx_ordering[procno::args.num_proc]
		param_batches.append((idx_batch, slices.copy(), args))

	with poolcontext(args.num_proc) as p:
		proc_slices = p.map(get_confs_for_idxs_unpack, param_batches)

	slices = aggregate_proc_confs(proc_slices, slices, args.data_dir)

	# Save the calibration data

	output_fname = args.output_file
	if output_fname is None:
		output_fname = args.slice_file

	save_slices(output_fname, slices)
	
	if args.test:
		os.remove(args.data_dir)
