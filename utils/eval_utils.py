def get_depth_of_label(pred_label, slices):
	if pred_label <= len(slices[0]):
		# If pred_label is a terminal, it's parent could be a few levels up in the hierarchy.
		#
		# Therefore, iterate through the slices until the terminal label is in a node with more than one child.
		# Then return that depth plus one since the depth of the terminal is actually one lower.
		for i, slc in enumerate(slices):
			for node in slc:
				terms = node.terminals
				if pred_label in terms and len(terms) > 1:
					return len(slices)-i+1
	else:
		# Otherwise, iterate through the slices until the predicted label is within the current slice and return that depth.
		total_nodes = 0
		for i, slc in enumerate(slices):
			if pred_label <= len(slc) + total_nodes:
				return len(slices)-i

			total_nodes += len(slc)			

def is_in_gt_path(pred_label, gt_label, slices):
	total_nodes = 0
	for slc in slices:
		# Accumulate the total nodes before the current slice so that when gt_label is remapped
		# to the local indices of the slice, that base is added to test for equality with the predicted label.
		if pred_label <= len(slc) + total_nodes:
			gt_remapped = remap_label(gt_label, slc) + total_nodes
			return gt_remapped == pred_label
		
		total_nodes += len(slc)