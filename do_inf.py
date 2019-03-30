import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys

import util

def perform_inference_on_chunk(chunkno, chunklen, slices, args):
	batch_size = 10000 # calculate later	
	batch_sizes = [batch_size] * (chunklen//batch_size)
	batch_sizes[-1] -= chunklen % batch_size

	for batch in batch_sizes:
		shapes, num_pix, fg_masks, lgts, _ = util.unserialize_example(args.imset, batch, chunkno)	
		lgts = lgts.reshape(-1, util.nc)
		term_preds = np.argmax(lgts, -1) + 1

		per_slice_preds = []
		per_slice_confs = []

		if not args.sm_by_slice: scores = sm_of_logits(logits, zero_pad=True)
		else: scores = logits

		for slc in slices:
			slc_conf_mask = np.zeros_like(term_preds)
			slc_conf_map = np.zeros((len(lgts)), dtype=logits.dtype)

			slc_pred_labs = remap_label_arr(term_preds, slc)
			slc_scores = remap_scores_arr(scores, slc)

			if args.sm_by_slice: slc_sm = sm_of_logits(slc_scores)
			else: slc_sm = slc_scores

			for slc_pred_lab in np.unique(slc_pred_labs):
				node = slc[slc_pred_lab]

				pred_mask = slc_pred_labs == slc_pred_lab	
				slc_sm_masked = slc_sm[pred_mask]
				confs = node.conf_for_scoers(slc_sm_masked)

				pred_lab = node.node_idx
				if len(node.terminals) == 1:
					pred_lab = node.terminals[0]

				slc_conf_mask[pred_mask] = pred_lab
				slc_conf_map[pred_mask] = confs

			slc_conf_mask = set_fg_in_larger_array(slc_fg_conf_mask, fg_mask, orig_shape)
			per_slice_preds.append(slc_conf_mask)
			per_slice_confs.append(slc_conf_map)

		for i, (shape, n) in enumerate(zip(shapes, num_pix)):
			idx = chunkno * (args.n_img // args.num_proc) + i
			fg_mask = fg_masks[:n]

			confident_masks = []
			confidence_maps = []

			for slice_pred, slice_conf in zip(per_slice_preds, per_slice_confs):
				conf_mask = util.set_fg_in_larger_array(slice_pred[:n], fg_mask, shape)
				conf_map = util.set_fg_in_larger_array(slice_conf[:n], fg_mask, shape)
		
				confident_masks.append(conf_mask)
				confidence_maps.append(conf_map)	

			save_calib_pred(args.imset, idx, args.name, confident_masks, confidence_maps)
	
def perform_inference_on_chunk_unpack(params):
	perform_inference_on_chunk(*params)
	
if __name__ == '__main__':
	from argparse import ArgumentParser
	parser = ArgumentParser(description='Build the calibration hierarchy using multiprocessing.')
	parser.add_argument('--slice_file', dest='slice_file', type=str, default='slices.pkl', help='The pickle file that specifies the hierarchy.')
	parser.add_argument('--imset', dest='imset', type=str, default='test', help='The image set to build the calibration histograms from. Either val or test')
	parser.add_argument('--num_proc', dest='num_proc', type=int, default=8, help='The number of processes to spawn to parallelize calibration.')
	parser.add_argument('--sm_by_slice', dest='sm_by_slice', action='store_true', help='Whether or not to take the softmax of the logits at each slice of the hierarchy. True by default.')
	parser.add_argument('--name', dest='name', type=str, help='The name of the current method.')
	parser.add_argument('--test', dest='test', action='store_true', help='Whether or not to test the inference on a small subset of the dataset.')
	args = parser.parse_args()

	slices = util.read_slices(args.slice_file, reset=False)
	args.nb = len(slices[0][0].acc_hist)

	n_img = util.num_img_for(args.imset)
	args.n_img = n_img

	img_per_chunk = [n_img//args.num_proc] * args.num_proc
	img_per_chunk[-1] -= n_img % args.num_proc

	param_batches = [(i, chnk_len, slices.copy(), args) for i, chnk_len in zip(range(args.num_proc), img_per_chunk)] 

	with poolcontext(args.num_proc) as p:
		_ = p.map(perform_inference_on_chunk_unpack, param_batches)
