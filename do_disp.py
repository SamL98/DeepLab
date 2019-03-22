from hdf5storage import loadmat
from os.path import join
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

from util import *
from perform_calibrated_inference import calibrate_logits

from argparse import ArgumentParser
parser = ArgumentParser(description='Build the calibration hierarchy using multiprocessing.')
parser.add_argument('--slice_file', dest='slice_file', type=str, default='slices.pkl', help='The pickle file that specifies the hierarchy.')
parser.add_argument('--imset', dest='imset', type=str, default='test', help='The image set to build the calibration histograms from. Either val or test')
parser.add_argument('--conf_thresh', dest='conf_thresh', type=float, default=0.75, help='The confidence threshold for inference.')
parser.add_argument('--name', dest='name', type=str, help='The name of the current method.')
parser.add_argument('--idx', dest='idx', type=int, default=-1, help='The index of the result to display.')
args = parser.parse_args()

def load_pred(args):
	gt = load_gt(args.imset, args.idx, reshape=False)

	bg_mask = (1-fg_mask_for(gt)).astype(np.bool)
	gt[bg_mask] = 0

	logits = load_logits(args.imset, args.idx)
	logits[...,0] = 0

	dl_pred = np.argmax(logits, axis=-1)
	dl_pred[bg_mask] = 0
	
	calib_pred = load_calib_pred(args.imset, args.idx, args.conf_thresh, args.name)
	if calib_pred is None:
		exit()

	return gt, dl_pred, calib_pred

slices = read_slices(args.slice_file)

if args.idx < 0:
	m = num_img_for(args.imset)

	done = False
	while not done:
		args.idx = np.random.choice(m, 1)[0]
		gt, dl_pred, calib_pred = load_pred(args)
		done = not (set(np.unique(dl_pred)) == set(np.unique(calib_pred)))

	print(args.idx)
else:
	gt, dl_pred, calib_pred = load_pred(args)

img = load_rgb(args.imset, args.idx)

print('GT Labels:')
for lab in np.unique(gt):
	print('%d: %s' % (lab, classes[lab]))
print('***')

print('DL Labels:')
for lab in np.unique(dl_pred):
	print('%d: %s' % (lab, classes[lab]))
print('***')
	
print('CALIB Labels:')
for lab in np.unique(calib_pred):
	if lab == 0: 
		print('0: background')
		continue
		
	base = 0
	for slice in slices:
		if lab-base <= len(slice):
			print('%d: %s' % (lab, slice[lab-base-1].name))
			break
		base += len(slice)
print('***')

sys.stdout.flush()

max_label = calib_pred.max()

fig, ax = plt.subplots(2, 2)

ax[0,0].imshow(img)
ax[0,0].set_title('Input')
ax[0,0].axis('off')

ax[0,1].imshow(gt, 'jet', vmin=0, vmax=max_label)
ax[0,1].set_title('Ground Truth')
ax[0,1].axis('off')

ax[1,0].imshow(dl_pred, 'jet', vmin=0, vmax=max_label)
ax[1,0].set_title('DeepLab Pred')
ax[1,0].axis('off')

ax[1,1].imshow(calib_pred, 'jet', vmin=0, vmax=max_label)
ax[1,1].set_title('Calibrated Pred')
ax[1,1].axis('off')

fig.savefig('images/vis_calib_%s_%d_%s.png' % (args.imset, args.idx, args.name), bbox_inches='tight')
plt.show()