import sys
from os.path import join, isfile
from skimage.io import imread
from hdf5storage import loadmat
import matplotlib.pyplot as plt

imset = sys.argv[1].lower()
idx = int(sys.argv[2])

iou = None
if isfile('iou_%s_sorted.txt' % imset):
	with open('iou_%s_sorted.txt' % imset) as f:
		for line in f:
			terms = line.split(': ')
			if int(terms[0]) == idx:
				iou = float(terms[1])
				break

root = 'D:/datasets/processed/voc2012'
rgb = imread(join(root, 'rgb', imset, '%s_%06d_img.png' % (imset, idx)))
gt = loadmat(join(root, 'truth', imset, '%s_%06d_pixeltruth.mat' % (imset, idx)))['truth_img']
pred = loadmat(join(root, 'deeplab_prediction', imset, '%s_%06d_prediction.mat' % (imset, idx)))['pred_img']

_, ax = plt.subplots(1, 3)

ax[0].imshow(rgb)
ax[0].set_title('%s %d' % (imset, idx))

ax[1].imshow(gt, 'jet')
ax[1].set_title('Ground Truth')

ax[2].imshow(pred, 'jet')
if iou is None:
	ax[2].set_title('Predicted')
else:
	ax[2].set_title('IoU: %f' % iou)

plt.show()