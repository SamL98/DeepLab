import numpy as np
import hdf5storage
from PIL import Image
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


HOME_DIR = '/Users/samlerner/Projects/refinenet/datasets/PascalContext/' # path to dataset (w/ RGB and Truth dirs)
RGB_DIR = HOME_DIR + 'RGB/'
OUTPUT_DIR = '/Users/samlerner/Projects/deeplab/cache_data/output' # where you want to the output (Train and Val dirs)
PATH_TO_DEEPLAB_GRAPH = "/Users/samlerner/Projects/deeplab/model_trained/deeplab_pascal/frozen_inference_graph.pb" # path to the xception65 saved model

NUM_IMAGE_CHANNELS = 3

def save_deeplab_features(rgb_dir, deeplab_output_dir, train_or_val, num_to_process):

	# -----------------------------------------------------------------------

	# reset graph
	tf.reset_default_graph()

	with tf.gfile.GFile(PATH_TO_DEEPLAB_GRAPH, 'rb') as f:
		graph_def = tf.GraphDef.FromString(f.read())	

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, name='')

		# for op in graph.get_operations():
		# 	print(op.name)
		# exit()

	# -----------------------------------------------------------------------

	with tf.Session(graph=graph) as sess:

		# iter through all images
		pix_mean = np.reshape([128, 128, 128], (1, 1, 3))
		for ind in range(num_to_process):

			# load rgb image
			if train_or_val=='train':
				filename = rgb_dir + 'Train/train_%06d_img.png' % (ind + 1)
			elif train_or_val=='val':
				filename = rgb_dir + 'Val/val_%06d_img.png' % (ind + 1)
			print('Loading', filename)
			rgb_img = Image.open(filename)
			rgb_img = np.array(rgb_img, dtype=np.float32)

			# mean-subtract the image
			#rgb_img = np.subtract(rgb_img, pix_mean)

			# make into a batch=1 format (batch=1 x rows x cols x depth)
			rgb_img = np.expand_dims(rgb_img, 0)

			# -----------------------------------------------------------------------

			# get deeplab features for image
			deeplab_features = graph.get_tensor_by_name('decoder/concat:0') # concatenation of encoder features and low-level features after resize, before dual 3x3 convolution
			#deeplab_features = graph.get_tensor_by_name('decoder/decoder_conv1_pointwise/Relu:0') # concatenation of encoder features and low-level features AFTER dual 3x3 convolution and before 1x1,num_classes convolution

			crop_size = 513
			num_feat = tf.shape(deeplab_features)[3]

			deeplab_features = tf.image.resize_images(
				deeplab_features,
				[crop_size, crop_size],
				method=tf.image.ResizeMethod.BILINEAR,
				align_corners=True
			)

			h, w = rgb_img.shape[1], rgb_img.shape[2]

			# we only need to remove the padding if one of the original dimensions is smaller than the crop size
			if h < crop_size or w < crop_size:
				deeplab_features = tf.slice(
					deeplab_features,
					[0, 0, 0, 0],
					[1, min(h, crop_size), min(w, crop_size), num_feat]
				)
			
			# we only need to resize to the original size if one of the original dimensions is larger than the crop size
			if h > crop_size or w > crop_size:
				deeplab_features = tf.image.resize_images(
					deeplab_features,
					[h, w],
					method=tf.image.ResizeMethod.BILINEAR,
					align_corners=True
				)

			deeplab_features = tf.squeeze(deeplab_features)
			clarinet_features = sess.run(deeplab_features, feed_dict={'ImageTensor:0': rgb_img})

			# -----------------------------------------------------------------------

			# save deeplab features
			if train_or_val=='train':
				filename = OUTPUT_DIR + 'Train/train_%06d_features.mat' % (ind + 1)
			elif train_or_val=='val':
				filename = OUTPUT_DIR + 'Val/val_%06d_features.mat' % (ind + 1)
			print('\tSaving', filename)
			hdf5storage.savemat(filename, {'clarinet_features': clarinet_features})
			
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


