import tensorflow as tf
import numpy as np
from PIL import Image
from hdf5storage import loadmat, savemat
from os.path import join

import util
import augment as aug
	
import sys
imset = sys.argv[1].lower()

GRAPH_PATH = 'E:/lerner/deeplab/model_trained/deeplabv3_pascal_train_aug/frozen_inference_graph.pb' 
OUTPUT_TENSOR_NAME = 'ResizeBilinear_2:0' 
INPUT_TENSOR_NAME = 'ImageTensor:0'
CROP_SIZE = 513
PIX_MEAN = 128

def restore_graph():
	tf.reset_default_graph()

	with tf.gfile.GFile(GRAPH_PATH, 'rb') as f: 
		graph_def = tf.GraphDef.FromString(f.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, name='') 

	return graph
	
def get_logits(inpt_im, ckpt_graph):
	h, w, _ = inpt_im.shape
	pad_h = max(0, CROP_SIZE-h)
	pad_w = max(0, CROP_SIZE-w)

	inpt_im = np.pad(
		inpt_im,
		[(0, pad_h), (0, pad_w), (0, 0)],
		'constant',
		constant_values=PIX_MEAN)

	inpt_im = np.expand_dims(inpt_im, 0)
		
	with tf.Session(graph=ckpt_graph) as sess:
		graph = tf.get_default_graph()
		output_tensor = graph.get_tensor_by_name(OUTPUT_TENSOR_NAME)
		
		output_tensor = tf.image.resize_images(output_tensor,
											[CROP_SIZE, CROP_SIZE],
											method=tf.image.ResizeMethod.BILINEAR,
											align_corners=True)
		
		if h < CROP_SIZE or w < CROP_SIZE:
			output_tensor = tf.slice(output_tensor,
								[0, 0, 0, 0],
								[1, min(h, CROP_SIZE), min(w, CROP_SIZE), 21])
									
		if h > CROP_SIZE or w > CROP_SIZE:
			output_tensor = tf.image.resize_images(output_tensor,
											[h, w],
											method=tf.image.ResizeMethod.BILINEAR,
											align_corners=True)
		
		logits = sess.run(output_tensor, feed_dict={INPUT_TENSOR_NAME: inpt_im})
		
	return np.squeeze(logits)

if __name__ == '__main__':
	ckpt_graph = restore_graph()

	for im_idx in range(1, util.num_img_for(imset)+1):
		util.stdout_writeln('Performing inference on image %d' % im_idx)
	
		rgb = util.load_rgb(imset, im_idx)
		if rgb.shape[2] == 4:
			rgb = rgb[...,:-1]
			
		gt = util.load_gt(imset, im_idx)

		rgb_aug, flip_idxs = aug.generate_augmentation_volume(rgb)
		util.save_rgb_aug(imset, im_idx, rgb_aug, flip_idxs)

		
		logits_aug = np.zeros((rgb_aug.shape[0], rgb_aug.shape[1], rgb_aug.shape[2], util.nc+1), dtype=np.float32)

		for i in range(len(rgb_aug)):
			logits = get_logits(rgb_aug[i], ckpt_graph)
			if i in flip_idxs:
				logits = np.fliplr(logits)

			logits_aug[i] = logits.copy()

		util.save_lgt_aug(imset, im_idx, logits_aug)
