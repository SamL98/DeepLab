from util import *
import numpy as np

def validate_calibration(slice_file, num_fg_pix, per_class_counts):
	''' Validate item #1 '''
	slices = read_slices(slice_file)

	slc0_num_fg_pix = sum([node.tot_hist.sum() for node in slices[0]])
	for i, slc in enumerate(slices[1:]):
		slc_num_fg_pix = sum([node.tot_hist.sum() for node in slc])
		perc_diff = abs(slc_num_fg_pix-slc0_num_fg_pix) / float(slc0_num_fg_pix)
		if perc_diff > 0.1:
			print('Non-consistent num_fg_pix across slices: %d vs %d for slice %d' % (slc0_num_fg_pix, slc_num_fg_pix, i+1))

	''' Validate item #2 '''
	perc_diff = abs(num_fg_pix - slc0_num_fg_pix) / num_fg_pix
	if perc_diff > 0.1:
		print('Non-consistent num_fg_pix on slice 0: %d vs %d' % (num_fg_pix, slc0_num_fg_pix))

	''' Validate item #3 '''
	for slc in slices:
		for node in slc:
			init_num_pix = 0
			for lab in node.terminals:
				init_num_pix += per_class_counts[lab+1]

			final_num_pix = node.tot_hist.sum()
			perc_diff = abs(init_num_pix-final_num_pix)/init_num_pix
			if perc_diff > 0.1:
				print('Non-consistent num_pix for %s node: %d vs %d' % (node.name, init_num_pix, final_num_pix))

	''' Validate item #4 '''
	for slc in slices:
		for node in slc:
			conf_adj_acc_hist = node.get_conf_acc_hist()
			num_out_of_range = ((conf_adj_acc_hist < 0) | (conf_adj_acc_hist > 1)).sum()
			if num_out_of_range > 0:
				print('Confidence values out of range for %s node: %s' % (node.name, conf_adj_acc_hist.__repr__()))


def validate_inference(slice_file, imset, name):
	''' Validate item #1 '''
	slices = read_slices(slice_file)

	for idx in range(1, num_img_for(imset)+1):
		logits = load_logits(imset, idx, reshape=False)
		calib_pred = load_calib_pred(imset, idx, name, conf=0.75).ravel()
		dl_pred = (np.argmax(logits[:,1:], -1) + 1).ravel()

		for calib_lab, dl_lab in zip(calib_pred, dl_pred):
			if not is_in_gt_path(calib_lab, dl_lab, slices):
				print('Calibrated prediction jumped paths: %d -> %d' % (dl_lab, calib_lab))