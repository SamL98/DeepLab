from util import *

slices = read_slices('slices.pkl')

for idx in range(1, num_img_for('val')):
	logits = load_logits('val', idx, reshape=True)
	gt = load_gt('val', idx, reshape=True)

	fg_mask = fg_mask_for(gt)
	logits = logits[fg_mask]
	gt = gt[fg_mask]

	terminal_pred = np.argmax(logits[:,1:], axis=-1) + 1

	for i, slc in enumerate(slices):
		slc_gt = np.array([remap_label(lab, slc) for lab in gt])
		slc_term_pred = np.array([remap_label(pred, slc) for pred in terminal_pred])
		slc_sm = np.array([remap_scores(sm_vec, slc) for sm_vec in sm])

		node_idx = 1
		pred_mask = slc_term_pred == node_idx

		slc_gt_masked = slc_gt[pred_mask]
		slc_sm_masked = slc_sm[pred_mask]

		sm_conf = slc_sm_masked[:,node_idx]

		node.accum_scores(sm_conf, slc_gt_masked == node_idx, 100, 0.1)