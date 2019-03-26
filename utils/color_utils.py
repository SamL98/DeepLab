import numpy as np

def voc_colormap():
	colormap = np.zeros((256, 3), dtype=int)
	ind = np.arange(256, dtype=int)

	for shift in reversed(range(8)):
		for channel in range(3):
			colormap[:, channel] |= ((ind >> channel) & 1) << shift
		ind >>= 3

	return colormap
	
def cvt_to_rgb(labelmap):
	colormap = voc_colormap()
	return colormap[labelmap]