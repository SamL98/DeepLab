import numpy as np
from hdf5storage import loadmat, savemat
from os.path import join
import sys

import util

def perform_inference_on_chunk(chunkno, slices, args):
	batch_size = 100 # calculate later
	max_dim = 512

	lgts = np.zeros((batch_size * max_dim**2, util.nc), dtype=util.DTYPES[util.LOGITS])
	gt = np.zeros((batch_size * max_dim**2), dtype=util.DTYPES[util.GT])
	fg = np.zeros((batch_size * max_dim**2), dtype=util.DTYPES[util.FG])
	shapes = np.zeros((batch_size, 2), dtype=util.DTYPES[util.SHAPE])
	
	done = False
	base_img_idx = chunkno * (args.n_img // args.num_proc)

	while not done:
		done, num_img, num_pix, num_fg_pix = util.unserialize_examples_for_inf(args.imset, batch_size, chunkno, lgts, gt, fg, shapes)	
		util.stdout_writeln(f'Nimg: {num_img}')
		util.stdout_writeln(f'Npix: {num_pix}')
		util.stdout_writeln(f'Nfgpix: {num_fg_pix}')

		batch_fg = fg[:num_pix]
		batch_lgts = lgts[:num_fg_pix]
		batch_term_preds = np.argmax(batch_lgts, -1)
		batch_gt = gt[:num_fg_pix]
		batch_gt -= 1
		
		if done:
			shapes = shapes[:num_img]

		per_slice_preds = []
		per_slice_confs = []

		sm = util.sm_of_logits(batch_lgts)

		for j, slc in enumerate(slices):
			slc_conf_mask = np.zeros_like(batch_term_preds)
			slc_conf_map = np.zeros((len(batch_lgts)), dtype=batch_lgts.dtype)

			sm = slc.remap_scores_and_labels(sm, batch_gt, batch_term_preds)

			for slc_pred_lab in np.unique(batch_term_preds):
				node = slc[slc_pred_lab]

				pred_mask = batch_term_preds == slc_pred_lab	
				confs = node.conf_for_scores(sm[pred_mask,slc_pred_lab])

				pred_lab = node.node_idx
				slice_idx = j
				while slice_idx > 0 and len(node.children) == 1:
					pred_lab = node.children[0]
					node = slices[slice_idx-1][pred_lab]
					slice_idx -= 1

				#util.stdout_writeln('sdflkjsdf')
				#util.stdout_writeln(pred_mask.shape.__repr__())
				#util.stdout_writeln(confs.shape.__repr__())

				slc_conf_mask[pred_mask] = pred_lab
				slc_conf_map[pred_mask] = confs

			per_slice_preds.append(slc_conf_mask)
			per_slice_confs.append(slc_conf_map)

		pix_accum = 0
			
		for i, shape in enumerate(shapes):
			h, w = shape
			num_pix = h*w
			idx = base_img_idx + i
			
			fg_mask = batch_fg[pix_accum:pix_accum+num_pix]

			confident_masks = []
			confidence_maps = []

			for slice_pred, slice_conf in zip(per_slice_preds, per_slice_confs):
				util.stdout_writeln('sdflkjsdf')
				util.stdout_writeln(slice_pred.shape.__repr__())
				util.stdout_writeln(slice_conf.shape.__repr__())
				util.stdout_writeln(fg_mask.shape.__repr__())
				util.stdout_writeln(shape.__repr__())
				conf_mask = util.set_fg_in_larger_array(slice_pred[pix_accum:pix_accum+num_pix], fg_mask, shape)
				conf_map = util.set_fg_in_larger_array(slice_conf[pix_accum:pix_accum+num_pix], fg_mask, shape)
		
				confident_masks.append(conf_mask)
				confidence_maps.append(conf_map)	

			save_calib_pred(args.imset, idx, args.name, confident_masks, confidence_maps)
			
			pix_accum += num_pix
			
		base_img_idx += num_img
	
def perform_inference_on_chunk_unpack(params):
	perform_inference_on_chunk(*params)
	
if __name__ == '__main__':
	from argparse import ArgumentParser
	parser = ArgumentParser(description='Build the calibration hierarchy using multiprocessing.')
	parser.add_argument('--slice_file', dest='slice_file', type=str, default='slices.pkl', help='The pickle file that specifies the hierarchy.')
	parser.add_argument('--imset', dest='imset', type=str, default='test', help='The image set to build the calibration histograms from. Either val or test')
	parser.add_argument('--num_proc', dest='num_proc', type=int, default=8, help='The number of processes to spawn to parallelize calibration.')
	parser.add_argument('--name', dest='name', type=str, help='The name of the current method.')
	parser.add_argument('--test', dest='test', action='store_true', help='Whether or not to test the inference on a small subset of the dataset.')
	args = parser.parse_args()

	slices = util.read_slices(args.slice_file)
	args.nb = len(slices[0][0].acc_hist)
	args.n_img = util.num_img_for(args.imset)

	slices = util.read_slices(args.slice_file)
	param_batches = [(i, slices.copy(), args) for i in range(args.num_proc)] 

	with util.poolcontext(args.num_proc) as p:
		_ = p.map(perform_inference_on_chunk_unpack, param_batches)
