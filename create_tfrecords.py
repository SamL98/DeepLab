import tensorflow as tf
from skimage.io import imread
import h5py
import numpy as np

from slim.datasets.dataset_utils import bytes_feature, int64_feature
from slim.datasets import dataset_utils
from deeplab.datasets.build_data import image_seg_to_tfexample

import os
from os.path import join
import sys

w, h = 400, 400
rec_dir = join('datasets', 'PascalContext', 'tfrecord')

def img_to_tfrecord(image_data, mask_data, width, height, image_format, filename):
	return tf.train.Example(features=tf.train.Features(feature={
		'image/encoded': bytes_feature(image_data),
		'image/filename': bytes_feature(bytes(filename, 'utf8')),
		'image/format': bytes_feature(image_format),
		'image/height': int64_feature(height),
		'image/width': int64_feature(width),
		'image/channels': int64_feature(3),
		'images/segmentation/class/encoded': bytes_feature(mask_data),
		'images/segmentation/class/format': bytes_feature(image_format)
	}))


import math

def _add_to_tfrecord(ext, num_images):
	print('Convert dataset ' + ext)

	dataset_dir = '/Users/samlerner/Projects/refinenet/datasets/PascalContext'
	img_dir = join(dataset_dir, 'RGB', ext)
	mask_dir = join(dataset_dir, 'Truth', ext)
	img_files = os.listdir(img_dir)

	dest_dir = '/Users/samlerner/Projects/deeplab/datasets/PascalContext'

	num_shards = 4
	num_per_shard = int(math.ceil(num_images / float(num_shards)))

	with tf.Graph().as_default():
		with tf.Session('') as sess:
			for shard in range(num_shards):
				print('Converting shard %d / %d' % (shard, num_shards))
				output_filename = join(dest_dir, 'tfrecord', '%s-%05d-of-%05d.tfrecord' % (ext.lower(), shard, num_shards))

				with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
					for j in range(shard*num_per_shard, min(num_images, (shard+1)*num_per_shard)):
						sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
						sys.stdout.flush()

						img = imread(join(img_dir, '%s_%06d_img.png' % (ext.lower(), j+1)))
						shape = img.shape

						image = tf.placeholder(dtype=tf.uint8, shape=shape)
						encoded_png = tf.image.encode_png(image)

						mask = tf.placeholder(dtype=tf.uint8, shape=(shape[0], shape[1], 1))
						encoded_mask = tf.image.encode_png(mask)

						mask_file = h5py.File(join(mask_dir, '%s_%06d_pixeltruth.mat' % (ext.lower(), j+1)))

						png_string = sess.run(encoded_png, feed_dict={image: img})
						mask_string = sess.run(encoded_mask, feed_dict={mask: np.expand_dims(mask_file['truth_img'][:,:].T, axis=2)})

						mask_file.close()

						#example = img_to_tfrecord(
						#	png_string, mask_string, w, h, 'png'.encode(), img_files[j])
						example = image_seg_to_tfexample(png_string, img_files[j], h, w, mask_string)
						tfrecord_writer.write(example.SerializeToString())



ds_info = h5py.File('/Users/samlerner/projects/refinenet/datasets/PascalContext/dataset_info.mat')
num_train, num_val, num_test = ds_info['num_train'][0,0], ds_info['num_val'][0,0], ds_info['num_test'][0,0]
ds_info.close()

dno = int(sys.argv[1])

if dno >= 4:
	dno -= 4
	_add_to_tfrecord('Train', num_train)

if dno >= 2:
	dno -= 2
	_add_to_tfrecord('Val', num_val)

if dno >= 1:
	_add_to_tfrecord('Test', num_test)
