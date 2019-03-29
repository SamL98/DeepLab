import numpy as np

def confident_mask(masks, conf_maps, conf_thresh):
	confident_mask = np.zeros_like(masks[0])

	for mask, conf_map in zip(reversed(masks), reversed(conf_maps)):
		conf_mask = conf_map >= conf_thresh
		confident_mask[conf_mask] = mask[conf_mask]

	return confident_mask