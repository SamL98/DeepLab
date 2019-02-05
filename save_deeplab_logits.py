import numpy as np
from skimage.transform import resize
import hdf5storage
from PIL import Image
import os
from os.path import join
import tensorflow as tf
import tensorflow.contrib.slim as slim


HOME_DIR = '/Users/samlerner/Projects/refinenet/datasets/PascalContext/' # path to dataset (w/ RGB and Truth dirs)
RGB_DIR = HOME_DIR + 'RGB/'
OUTPUT_DIR = '/Users/samlerner/Projects/deeplab/cache_data/output2' # where you want to the output (Train and Val dirs)
#PATH_TO_DEEPLAB_GRAPH = "/Users/samlerner/Projects/deeplab/model_trained/pc/frozen_inference_graph-2.pb" # path to the xception65 saved model
PATH_TO_DEEPLAB_CKPT = "/Users/samlerner/Projects/deeplab/model_trained/ckpt/"
CKPT_NAME = "model.ckpt-30000"

NUM_IMAGE_CHANNELS = 3
CROP_SIZE = 513
BATCH_SIZE = 3

def resize_clarinet_feature(feat, h, w):
	feat = resize(feat, (CROP_SIZE, CROP_SIZE), preserve_range=True)

	if h < CROP_SIZE or w < CROP_SIZE:
		feat = feat[:min(h, CROP_SIZE), :min(w, CROP_SIZE)]

	if h > CROP_SIZE or w > CROP_SIZE:
		feat = resize(feat, (h, w), preserve_range=True)

	return feat


def save_deeplab_features(rgb_dir, deeplab_output_dir, train_or_val, num_to_process):

	# -----------------------------------------------------------------------

	# reset graph
	tf.reset_default_graph()

	# -----------------------------------------------------------------------

	# with tf.Session(graph=graph) as sess:
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

		# restore the graph/weights from the checkpoint
		saver = tf.train.import_meta_graph(PATH_TO_DEEPLAB_CKPT+CKPT_NAME+'.meta')
		saver.restore(sess, PATH_TO_DEEPLAB_CKPT+CKPT_NAME)
		
		# mean to pad small images with
		pix_mean = 128

		for i in range(0, num_to_process, BATCH_SIZE):
			print('Processing batch %d' % i)

			# open all of the image files for the batch
			batch = [Image.open(join(RGB_DIR, train_or_val, '%s_%06d_img.png') % (train_or_val, idx+1))
					for idx in range(i, min(num_to_process, i+BATCH_SIZE))]

			# create the batch tensor
			# batch_size is 3 since that is the size the checkpoint expects
			inpt = np.zeros((BATCH_SIZE, CROP_SIZE, CROP_SIZE, NUM_IMAGE_CHANNELS), dtype=np.float32)
			sizes = []

			# if there aren't three images left, just keep track so we don't save those features
			num_in_batch = BATCH_SIZE
			if i+BATCH_SIZE >= num_to_process:
				num_in_batch = num_to_process - i

			for j, batch_img in enumerate(batch):
				im = np.array(batch_img, dtype=np.float32)
				sizes.append(im.shape)

				pad_h = max(0, CROP_SIZE - im.shape[0])
				pad_w = max(0, CROP_SIZE - im.shape[1])

				inpt[j,...] = np.pad(im, 
							[(0, pad_h), (0, pad_w), (0, 0)], 
							'constant', 
							constant_values=pix_mean)

			# -----------------------------------------------------------------------

			# get deeplab features for image
			graph = tf.get_default_graph()
			# for op in graph.get_operations():
			# 	if 'logits' in op.name: print(op.name)
			# exit()

			# concatenation of encoder features and low-level features after resize, before dual 3x3 convolution
			deeplab_features = graph.get_tensor_by_name('decoder/concat:0')

			# concatenation of encoder features and low-level features AFTER dual 3x3 convolution and before 1x1,num_classes convolution
			#deeplab_features = graph.get_tensor_by_name('decoder/decoder_conv1_pointwise/Relu:0')

			clarinet_features = sess.run(deeplab_features, feed_dict={'image:0': inpt})

			# -----------------------------------------------------------------------

			# save deeplab features for each image in the batch individually
			for j in range(num_in_batch):
				print('Saving feature %d of batch %d' % (j, i))
				clarinet_feature, size = clarinet_features[j], sizes[j]
				#clarinet_feature = resize_clarinet_feature(clarinet_feature, size[0], size[1])
				hdf5storage.savemat(join(OUTPUT_DIR+train_or_val.capitalize(), '%s_%06d_features.mat' % (train_or_val, i+j+1)), 
									{'clarinet_features': clarinet_feature})

			# -----------------------------------------------------------------------

	print('Finished with', train_or_val)


def save_deeplab_features_wrapper():

	# load metadata
	filename = HOME_DIR + 'dataset_info.mat'
	print('Loading:', filename)
	mat_contents = hdf5storage.loadmat(filename)
	num_train = mat_contents['num_train']
	num_val = mat_contents['num_val']

	# make directories
	if not os.path.exists(OUTPUT_DIR + 'Train/'):
		os.makedirs(OUTPUT_DIR + 'Train/')
	if not os.path.exists(OUTPUT_DIR + 'Val/'):
		os.makedirs(OUTPUT_DIR + 'Val/')

	# save for train
	save_deeplab_features(RGB_DIR, OUTPUT_DIR, 'train', num_train)

	# save for val
	save_deeplab_features(RGB_DIR, OUTPUT_DIR, 'val', num_val)


if __name__ == '__main__':

	# run the code
	save_deeplab_features_wrapper()


