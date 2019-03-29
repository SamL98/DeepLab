import util
from os.path import join
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets

def stub_anno(a):
	anno = a.annotate('',
					  xy=(0, 0), 
					  xytext=(-20, 20), 
					  textcoords='offset points', 
					  bbox=dict(boxstyle='round', fc='w'), 
					  arrowprops=dict(arrowstyle='->'))
	anno.set_visible(False)
	return anno

def update_anno(anno, x, y, class_lab):
	anno.xy = (x, y)
	anno.set_text(class_lab)
	anno.get_bbox_patch().set_alpha(0.65)
	anno.set_visible(True)

def update_conf(*args):
	global fig, sconf, calib_mask_ax, masks, conf_maps, conf_mask, cmap
	conf = sconf.val
	conf_mask = util.confident_mask(masks, conf_maps, conf)
	calib_mask_ax.set_data(cmap[conf_mask])
	fig.canvas.draw_idle()

def hover(e):
	global ax
	global gt, dl_pred, conf_mask
	global last_lab, class_labs
	global gt_anno, dl_anno, calib_anno

	for i, j in [(0, 1), (1, 0), (1, 1)]:
		a = ax[i, j]
		mask = None
		anno = None

		if i == 0 and j == 1:
			mask = gt
			anno = gt_anno
		elif i == 1 and j == 0:
			mask = dl_pred
			anno = dl_anno
		else:
			mask = conf_mask
			anno = calib_anno

		if not a.contains(e)[0]:
			anno.set_visible(False)
			continue

		y, x = int(e.ydata), int(e.xdata)
		lab = mask[y, x]

		update_anno(anno, x, y, class_labs[lab])
	
	fig.canvas.draw_idle()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='View calibration results with confidence slider.')
	parser.add_argument('--name', dest='name', type=str)
	parser.add_argument('--idx', dest='idx', type=int, default=None)
	args = parser.parse_args()

	if not 'name' in args:
		exit()

	if not args.idx:
		args.idx = np.random.choice(list(util.imset_iter('test')), 1)[0]
		print(args.idx)

	cmap = util.voc_colormap()
	rgb = util.load_rgb('test', args.idx)
	gt = util.load_gt('test', args.idx)
	masks, conf_maps = util.load_calib_pred('test', args.idx, args.name)

	conf = 0
	conf_mask = util.confident_mask(masks, conf_maps, conf)
	dl_pred = conf_mask.copy()

	last_lab = -1
	num_lab = 0
	class_labs = ['background']
	for slc in util.read_slices(join('calib_data', args.name, 'slices.pkl')):
		for node in slc:
			class_labs.append(node.name)
			num_lab += 1
	class_labs = class_labs + ['VOID']

	cmap = np.concatenate((cmap[:num_lab+1], np.expand_dims(cmap[-1], 0)), axis=0)
	gt[gt==255] = len(cmap)-1

	fig, ax = plt.subplots(2, 2)

	ax[0,0].imshow(rgb)
	ax[0,0].axis('off')
	ax[0,0].set_title('Original Image')

	ax[0,1].imshow(cmap[gt])
	ax[0,1].axis('off')
	ax[0,1].set_title('Ground Truth')

	ax[1,0].imshow(cmap[dl_pred])
	ax[1,0].axis('off')
	ax[1,0].set_title('DeepLab Prediction')

	calib_mask_ax = ax[1,1].imshow(cmap[conf_mask])
	ax[1,1].axis('off')
	ax[1,1].set_title('Calibrated Prediction')

	gt_anno = stub_anno(ax[0,1])
	dl_anno = stub_anno(ax[1,0])
	calib_anno = stub_anno(ax[1,1])

	axconf = plt.axes([0.125, 0.04, 0.775, 0.04])
	sconf = mwidgets.Slider(axconf, 'Conf', 0, 1, valinit=conf)
	sconf.on_changed(update_conf)

	fig.canvas.mpl_connect('motion_notify_event', hover)
	plt.show()