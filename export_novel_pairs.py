import matplotlib.pyplot as plt
from skimage.io import imread
from hdf5storage import loadmat

root_dir = 'D:/datasets/processed/voc2012'
rgb_dir = '%s/rgb/novel_test_images' % root_dir
pred_dir = '%s/deeplab_prediction/novel_test_images' % root_dir
fmt = 'novel_test_images_%06d'

dst_dir = 'C:/users/lerner.67a/documents/novel_images'

num_img = 15
for i in range(1, num_img+1):
	rgb = imread('%s/%s_img.png' % (rgb_dir, fmt % i))
	pred = loadmat('%s/%s_prediction.mat' % (pred_dir, fmt % i))['pred_img']
	
	fig, ax = plt.subplots(1, 2)
	
	ax[0].imshow(rgb)
	ax[0].set_title('Original')
	
	ax[1].imshow(pred, 'jet')
	ax[1].set_title('Predicted')
	
	fig.savefig('%s/pair_%d.png' % (dst_dir, i))
	plt.close()