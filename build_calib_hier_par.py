import numpy as np
import multiprocessing as mp
from contextlib import contextmanager
from hdf5storage import loadmat, savemat
from os.path import join, isfile

from util import *

# Pool the a function across multiple inputs and wait for them to complete
@contextmanager
def poolcontext(num_proc):
    pool = mp.Pool(num_proc)
    yield pool
    pool.terminate()

# Get the histograms for the given logits and ground truth labels
def hists_for_pixels(logits, gt, slices, args, res, nb):
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
			# Since we are measuring precision in the calibration histograms, mask the ground truth and softmax by where the current node is softmax
			pred_labels = np.argmax(slc_sm, axis=-1)
			argmax_mask = pred_labels == j

			slc_gt_masked = slc_gt[argmax_mask]
			slc_sm_masked = slc_sm[argmax_mask]

			# Because of the previous mask, the j-th softmax value will always be the max
			sm_conf = slc_sm_masked[:,j]
			bins = np.floor(sm_conf/res).astype(np.uint8)

			# If sm_conf happend to be 1, then the bin will be nb which will cause an IndexError
			bins = np.minimum(bins, nb-1)

			for binno in np.unique(bins):
				# Mask the bin vector and ground truth by the current bin number
				bin_mask = bins == binno
				
				# We are already assured that j was predicted so the corre_hist is accumumlated by how many times we were right
				node.corr_hist[binno] += (slc_gt_masked[bin_mask] == j).sum()

				# The count hist is simply how many times this bin was measured
				node.count_hist[binno] += bin_mask.sum()

	return slices


# Return the correct and count histograms given the hierarchy specified by slices
def get_hists_for_idxs(idxs, slices, args):
	nb = len(slices[0][0].count_hist)
	res = 1./nb

	if args.load_to_memory:
		# If we are loading all of the pixels into memory, creating arrays for the logits and ground truth
		# labels that have the capacity of as many pixels there are in the image set
		num_pixel = len(idxs) * img_size**2
		logits = np.zeros((num_pixel, nc), dtype=np.float32)
		gt = np.zeros((num_pixel), dtype=np.uint8)

		# Load all of the logits and labels for each image into the buffers
		num_fg_pixels = 0
		for idx in idxs:
			im_logits = load_logits(args.imset, idx, reshape=True)
			im_gt = load_gt(args.imset, idx, reshape=True)

			fg_mask = fg_mask_for(gt)
			im_logits = im_logits[fg_mask]
			im_gt = im_gt[fg_mask]

			logits[num_fg_pixels:num_fg_pixels+len(im_logits)] = im_logits[:]
			gt[num_fg_pixels:num_fg_pixels+len(im_gt)] = im_gt[:]
			num_fg_pixels += fg_mask.sum()

		# Only keep the pixels that we stored into the buffer
		logits = logits[:num_fg_pixels]
		gt = gt[:num_fg_pixels]

		# Compute the histograms on these arrays -- it should be mostly vectorized
		slices = hists_for_pixels(logits, gt, slices, args, res, nb)
	else:
		# If we are computing the histograms on the fly, load each image individually and accumulate all the histograms
		for idx in idxs:
			logits = load_logits(args.imset, idx, reshape=True)
			gt = load_gt(args.imset, idx, reshape=True)

			fg_mask = fg_mask_for(gt)
			logits = logits[fg_mask]
			gt = gt[fg_mask]

			slices = hists_for_pixels(logits, gt, slices, args, res, nb)

	return slices

def get_hists_for_idxs_unpack(params):
	return get_hists_for_idxs(*params)

def aggregate_proc_hists(proc_slices, slices):
	for i, slc in enumerate(slices):
		for j, node in enumeratae(slc):
			for proc_slice in proc_slices:
				node.corr_hist[:] += proc_slice[i][j].corr_hist[:]
				node.count_hist[:] += proc_slice[i][j].count_hist[:]

	return slices


from argparse import ArgumentParser
parser = ArgumentParser(description='Build the calibration hierarchy using multiprocessing.')
parser.add_argument('--slice_file', dest='slice_file', type=str, default='slices.pkl', help='The pickle file that specifies the hierarchy.')
parser.add_argument('--imset', dest='imset', type=str, default='val', help='The image set to build the calibration histograms from. Either val or test')
parser.add_argument('--num_proc', dest='num_proc', type=int, default=1, help='The number of processes to spawn to parallelize calibration.')
parser.add_argument('--output_file', dest='output_file', type=str, default=None, help='The pickle file to output the calibration hierarchy to. None if slice_file to be overwritten.')
parser.add_argument('--dont_reset', dest='reset', action='store_false', help='Pass if you want to accumulate calibration histograms. Normally they are reset when this script is run.')
parser.add_argument('--sm_by_slice', dest='sm_by_slice', action='store_true', help='Whether or not to take the softmax of the logits at each slice of the hierarchy. True by default.')
parser.add_argument('--load_to_memory', dest='load_to_memory', action='store_true', help='Whether or not to store the batches into memory (you need a lot).')


if __name__ == '__main__':
	args = parser.parse_args()

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
	

	# Split the indexes up between processed to try and spread the work evenly

	idx_ordering = np.array(idx_ordering)
	param_batches = []

	for procno in range(args.num_proc):
		idx_batch = idx_ordering[procno::args.num_proc]
		param_batches.append((idx_batch, slices.copy(), args))

	with poolcontext(args.num_proc) as p:
		proc_slices = p.map(get_hists_for_idxs_unpack, param_batches)

	slices = aggregate_proc_hists(proc_slices, slices)

	# Finally, set the accuracy (precision) histograms based off the correct and count histograms

	for slc in slices:
		for node in slc:
			node.get_acc_hist()

	# Save the calibration histograms

	output_fname = args.output_file
	if output_fname is None:
		output_fname = args.slice_file

	save_slices(output_fname, slices)
