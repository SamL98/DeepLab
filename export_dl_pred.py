import tensorflow as tf
import numpy as np
from PIL import Image
from skimage.transform import resize
from hdf5storage import loadmat, savemat
from os.path import join

from convert_to_tree_labels import make_mapping, convert_terminal_img_to_tree_labels as convert_to_tree_labelmap

def resize_func(arr, h, w):
	arr = resize(arr, (CROP_SIZE, CROP_SIZE), preserve_range=True, order=1, anti_aliasing=False)

	if h < CROP_SIZE or w < CROP_SIZE:
		arr = arr[:min(h, CROP_SIZE), :min(w, CROP_SIZE)]
		
	if h > CROP_SIZE or w > CROP_SIZE:
		arr = resize(arr, (h, w), preserve_range=Tru, order=1, anti_aliasing=False)
		
	return arr
	
import sys
imset = sys.argv[1].lower().capitalize()

GRAPH_PATH = 'E:/lerner/deeplab/model_trained/deeplabv3_pascal_train_aug/frozen_inference_graph.pb' 
DS_PATH = 'D:/datasets/processed/voc2012' 
RGB_PATH = join(DS_PATH, 'rgb', imset)
OUTPUT_PATH = join(DS_PATH, 'Deeplab_Prediction', imset)

INPT_FMT = imset.lower()+'_%06d_rgb.jpg' 
PRED_MAT_NAME = 'pred_img'
PRED_FMT = imset.lower()+'_%06d_prediction.mat'

OUTPUT_TENSOR_NAME = 'ResizeBilinear_2:0' 
INPUT_TENSOR_NAME = 'ImageTensor:0'
CROP_SIZE = 513
PIX_MEAN = 128

tf.reset_default_graph()

with tf.gfile.GFile(GRAPH_PATH, 'rb') as f: 
	graph_def = tf.GraphDef.FromString(f.read())

with tf.Graph().as_default() as graph:
	tf.import_graph_def(graph_def, name='') 
	
ds_info = loadmat(join(DS_PATH, 'dataset_info.mat'))

if imset.lower() == 'val':
	num_img = 350
else:
	num_img = 1449-350
	
orig_labelmap = ds_info['class_labels']
orig_labelmap[orig_labelmap.index('potted plant')] = 'pottedplant'
tree_labelmap = loadmat(join(DS_PATH, 'Decision_Tree', 'concept_tree.mat'))['all_node_labels']
mapping = make_mapping(orig_labelmap, tree_labelmap)

ckpt_graph = graph
#with tf.Session(graph=graph) as sess: 
if __name__ == '__main__':
	for im_idx in range(1, num_img+1):
		print('Performing inference on image %d' % im_idx)
	
		inpt_im = np.array(Image.open(join(RGB_PATH, INPT_FMT % im_idx)), dtype=np.float32)
		if inpt_im.shape[2] == 4:
			inpt_im = inpt_im[:,:,:-1]
		h, w, _ = inpt_im.shape

		pad_h = max(0, CROP_SIZE-h)
		pad_w = max(0, CROP_SIZE-w)

		inpt_im = np.pad(
			inpt_im,
			[(0, pad_h), (0, pad_w), (0, 0)],
			'constant',
			constant_values=PIX_MEAN)
		
		inpt_im = np.expand_dims(inpt_im, axis=0)

		with tf.Session(graph=ckpt_graph) as sess:
			graph = tf.get_default_graph()
			output_tensor = graph.get_tensor_by_name(OUTPUT_TENSOR_NAME)
		
			output_tensor = tf.image.resize_images(output_tensor,
												[CROP_SIZE, CROP_SIZE],
												method=tf.image.ResizeMethod.BILINEAR,
												align_corners=True)
			#output_tensor.set_shape([1, CROP_SIZE, CROP_SIZE, 21])
		
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
		
		logits = np.squeeze(logits)
		pred = np.argmax(logits[...,1:], axis=-1)
		
		savemat(join(OUTPUT_PATH, PRED_FMT % im_idx),
				{PRED_MAT_NAME: pred})
